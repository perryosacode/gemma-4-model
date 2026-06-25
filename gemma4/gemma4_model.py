"""Implementation for Gemma4 (text) architecture.

Only the text decoder is implemented (vision / audio towers are ignored), mirroring how
Gemma3 was added.  Gemma4 introduces two features that the standard MLC ``PagedKVCache``
cannot express directly:

* Per-layer-type attention geometry.  *Sliding* layers use ``head_dim=256`` /
  ``kv_heads=8`` / ``rope_theta=10000`` / full rotary, while *global* layers use
  ``head_dim=512`` / ``kv_heads=2`` / ``rope_theta=1e6`` / partial rotary (128 of 512)
  and share K with V (``attention_k_eq_v``).
* A Mixture-of-Experts block that runs in parallel with a dense MLP every layer.

The ``PagedKVCache`` only allows ``attn_kind`` to vary per layer.  To stay within it we
create a single cache at the *maximum* geometry (``head_dim=512``, ``kv_heads=8``) with
``RopeMode.NONE`` and apply RoPE manually per layer.  Sliding layers zero-pad Q/K/V from
256 to 512 and global layers replicate their 2 KV heads to 8 (preserving the GQA mapping).
"""

import dataclasses
from typing import Any, Dict, List, Optional  # noqa: UP035

from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.gemma.gemma_model import GemmaEmbedding
from mlc_llm.model.model_utils import index_last_token
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Gemma4TextConfig(ConfigBase):
    """Configuration of the text model inside Gemma4."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262_144
    hidden_activation: Optional[str] = "gelu_pytorch_tanh"
    sliding_window: int = 1024
    layer_types: Optional[List[str]] = None  # noqa: UP006
    final_logit_softcapping: Optional[float] = 30.0
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704
    enable_moe_block: bool = True
    use_double_wide_mlp: bool = False
    hidden_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    rope_parameters: Optional[Dict[str, Any]] = None  # noqa: UP006
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    sliding_window_size: Optional[int] = None
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):  # noqa: PLR0912
        # --- RoPE: theta + partial rotary factor differ per layer type ---
        self.rope_theta_sliding = 10_000.0
        self.rope_theta_global = 1_000_000.0
        self.partial_rotary_factor_global = 0.25
        rope_params = self.rope_parameters or self.kwargs.get("rope_parameters", None)
        if rope_params:
            sliding = rope_params.get("sliding_attention") or {}
            full = rope_params.get("full_attention") or {}
            self.rope_theta_sliding = float(sliding.get("rope_theta", self.rope_theta_sliding))
            self.rope_theta_global = float(full.get("rope_theta", self.rope_theta_global))
            self.partial_rotary_factor_global = float(
                full.get("partial_rotary_factor", self.partial_rotary_factor_global)
            )

        # --- Layer types (sliding / full attention) ---
        if self.layer_types is None:
            self.layer_types = self.kwargs.get("layer_types", None)
        if self.layer_types is None:
            pattern = 6  # default 5:1 sliding:full
            self.layer_types = [
                "sliding_attention" if (i + 1) % pattern else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"

        # --- Guard against features that are present in the architecture but not yet
        #     implemented here (and disabled in the gemma-4-26B-A4B-it config). ---
        if self.hidden_size_per_layer_input:
            raise ValueError(
                "Gemma4 per-layer embeddings (hidden_size_per_layer_input>0) are not supported."
            )
        if self.num_kv_shared_layers:
            raise ValueError("Gemma4 KV sharing (num_kv_shared_layers>0) is not supported.")
        if self.use_double_wide_mlp:
            raise ValueError("Gemma4 use_double_wide_mlp is not supported.")
        if self.hidden_activation not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError("Only GeLU is supported as the activation for gemma4.")
        if self.tensor_parallel_shards != 1:
            raise ValueError("Gemma4 currently only supports tensor_parallel_shards=1.")

        if self.sliding_window_size is None:
            self.sliding_window_size = self.sliding_window

        # --- Context / prefill sizing (follows the pragmatic Gemma3 behaviour). ---
        if self.context_window_size == 0:
            self.context_window_size = self.kwargs.get("max_position_embeddings", 8192)
        # NOTE: cap the context window to the sliding window (or a sane default), as global
        # layers' unbounded context is approximated like Gemma3's working path.
        self.context_window_size = max(self.sliding_window_size, 8192)
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 8192)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 8192)


@dataclasses.dataclass
class Gemma4Config(ConfigBase):
    """Configuration of the Gemma4 model (wraps the text config)."""

    text_config: Optional[Gemma4TextConfig] = None
    vocab_size: int = 262_144
    tie_word_embeddings: bool = True
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    sliding_window_size: int = -1
    prefill_chunk_size: int = -1
    is_text_model: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        if self.text_config is None:
            self.is_text_model = True
            self.text_config = Gemma4TextConfig.from_dict(self.kwargs)

        if isinstance(self.text_config, Gemma4TextConfig):
            text_config_dict: Dict[str, Any] = dataclasses.asdict(self.text_config)  # noqa: UP006
        else:
            text_config_dict = dict(self.text_config)
        for k, v in text_config_dict.pop("kwargs", {}).items():
            text_config_dict[k] = v
        text_config_dict["tensor_parallel_shards"] = self.tensor_parallel_shards
        self.text_config = Gemma4TextConfig.from_dict(text_config_dict)

        self.vocab_size = self.text_config.vocab_size
        for k in ["context_window_size", "prefill_chunk_size", "sliding_window_size"]:
            if getattr(self, k) <= 0 and hasattr(self.text_config, k):
                setattr(self, k, getattr(self.text_config, k))


def _rms_norm_no_scale(x: Tensor, eps: float) -> Tensor:
    """RMSNorm without a learnable weight (Gemma4's ``with_scale=False`` norms).

    Computed in float32 to match the HuggingFace reference.
    """
    dtype = x.dtype
    xf = x.astype("float32")
    dim = xf.shape[-1]
    variance = op.sum(op.square(xf), axis=-1, keepdims=True) / dim
    out = xf / op.sqrt(variance + eps)
    return out.astype(dtype)


def _apply_rope(
    x: Tensor, positions: Tensor, *, head_dim: int, rotary_dim: int, theta: float
) -> Tensor:
    """NeoX rotate-half RoPE applied manually (the cache runs in ``RopeMode.NONE``).

    Dimension ``d`` is paired with ``d + head_dim // 2``; only the first ``rotary_dim``
    dimensions are rotated (the rest pass through, giving the global layers' partial
    rotary).  ``positions`` is the per-token absolute position from
    ``PagedKVCache.get_query_positions``; tokens are laid out row-major ``[batch, seq]``.
    """
    half = head_dim // 2
    n_rot = rotary_dim // 2

    def _te(x_te: te.Tensor, pos_te: te.Tensor) -> te.Tensor:
        dtype = x_te.dtype
        seq_len = x_te.shape[1]

        def compute(b: tirx.Var, s: tirx.Var, h: tirx.Var, d: tirx.Var):
            freq_idx = tirx.if_then_else(d < half, d, d - half)
            exponent = (freq_idx.astype("float32") * 2.0) / float(head_dim)
            pos = pos_te[b * seq_len + s].astype("float32")
            freq = pos / tirx.power(tirx.const(float(theta), "float32"), exponent)
            freq = tirx.if_then_else(freq_idx < n_rot, freq, tirx.const(0.0, "float32"))
            cos = tirx.cos(freq).astype(dtype)
            sin = tirx.sin(freq).astype(dtype)
            partner = tirx.if_then_else(d < half, d + half, d - half)
            sign = tirx.if_then_else(
                d < half, tirx.const(-1.0, dtype), tirx.const(1.0, dtype)
            )
            return x_te[b, s, h, d] * cos + sign * x_te[b, s, h, partner] * sin

        return te.compute(x_te.shape, compute, name="gemma4_rope")

    return op.tensor_expr_op(_te, "gemma4_rope", [x, positions])


class Gemma4Attention(nn.Module):
    """Self attention for one Gemma4 layer (sliding or global geometry)."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        tc = config.text_config
        self.eps = tc.rms_norm_eps
        self.num_q_heads = tc.num_attention_heads
        self.is_global = tc.layer_types[layer_idx] == "full_attention"

        if self.is_global:
            self.head_dim = tc.global_head_dim
            self.num_kv_heads = tc.num_global_key_value_heads
            self.rope_theta = tc.rope_theta_global
            self.rotary_dim = int(tc.global_head_dim * tc.partial_rotary_factor_global)
            self.k_eq_v = True
        else:
            self.head_dim = tc.head_dim
            self.num_kv_heads = tc.num_key_value_heads
            self.rope_theta = tc.rope_theta_sliding
            self.rotary_dim = tc.head_dim
            self.k_eq_v = False

        # Unified cache geometry (max over all layers).
        self.cache_head_dim = tc.global_head_dim
        self.cache_kv_heads = tc.num_key_value_heads

        self.q_proj = nn.Linear(tc.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(tc.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(
                tc.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, tc.hidden_size, bias=False)
        # q_norm / k_norm are scaled RMSNorm; v_norm is no-scale (handled inline).
        self.q_norm = nn.RMSNorm(self.head_dim, -1, self.eps, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, -1, self.eps, bias=False)

    def forward(
        self, hidden_states: Tensor, positions: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ):
        b, s, _ = hidden_states.shape
        hd, ch = self.head_dim, self.cache_head_dim

        q = op.reshape(self.q_proj(hidden_states), (b, s, self.num_q_heads, hd))
        q = self.q_norm(q)
        q = _apply_rope(q, positions, head_dim=hd, rotary_dim=self.rotary_dim, theta=self.rope_theta)

        if self.k_eq_v:
            kv = op.reshape(self.k_proj(hidden_states), (b, s, self.num_kv_heads, hd))
            k = self.k_norm(kv)
            k = _apply_rope(
                k, positions, head_dim=hd, rotary_dim=self.rotary_dim, theta=self.rope_theta
            )
            v = _rms_norm_no_scale(kv, self.eps)  # K==V, V is not rotated
        else:
            k = op.reshape(self.k_proj(hidden_states), (b, s, self.num_kv_heads, hd))
            k = self.k_norm(k)
            k = _apply_rope(
                k, positions, head_dim=hd, rotary_dim=self.rotary_dim, theta=self.rope_theta
            )
            v = op.reshape(self.v_proj(hidden_states), (b, s, self.num_kv_heads, hd))
            v = _rms_norm_no_scale(v, self.eps)

        # --- Unify to cache geometry: zero-pad head_dim and replicate KV heads ---
        if hd < ch:
            pad = [0, 0, 0, 0, 0, 0, 0, ch - hd]
            q = op.pad(q, pad)
            k = op.pad(k, pad)
            v = op.pad(v, pad)
        if self.num_kv_heads < self.cache_kv_heads:
            rep = self.cache_kv_heads // self.num_kv_heads
            k = op.repeat(k, rep, axis=2)
            v = op.repeat(v, rep, axis=2)

        qkv = op.concat([q, k, v], dim=2)
        out = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_q_heads, sm_scale=1.0),
            (b, s, self.num_q_heads, ch),
        )
        if hd < ch:
            out = op.split(out, [hd], axis=-1)[0]
        out = op.reshape(out, (b, s, self.num_q_heads * hd))
        return self.o_proj(out)


class Gemma4MLP(nn.Module):
    """Dense feed-forward block (GeLU-tanh gated)."""

    def __init__(self, config: Gemma4Config):
        tc = config.text_config
        self.gate_proj = nn.Linear(tc.hidden_size, tc.intermediate_size, bias=False)
        self.up_proj = nn.Linear(tc.hidden_size, tc.intermediate_size, bias=False)
        self.down_proj = nn.Linear(tc.intermediate_size, tc.hidden_size, bias=False)

    def forward(self, x: Tensor):
        return self.down_proj(op.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4Router(nn.Module):
    """MoE router: norm + scale, softmax-over-all, top-k, renormalize, per-expert scale."""

    def __init__(self, config: Gemma4Config):
        tc = config.text_config
        self.hidden_size = tc.hidden_size
        self.eps = tc.rms_norm_eps
        self.top_k = tc.top_k_experts
        self.proj = nn.Linear(tc.hidden_size, tc.num_experts, bias=False)
        self.scale = nn.Parameter((tc.hidden_size,))
        self.per_expert_scale = nn.Parameter((tc.num_experts,))

    def forward(self, x: Tensor):
        # x: [num_tokens, hidden_size]
        h = _rms_norm_no_scale(x, self.eps)
        h = h * self.scale * (self.hidden_size**-0.5)
        scores = self.proj(h)
        # gating_softmax_topk does softmax-over-all -> top-k -> renormalize (norm_topk_prob).
        weights, indices = op_ext.moe_misc.gating_softmax_topk(scores, self.top_k)
        weights = weights * op.take(self.per_expert_scale, indices, axis=0)
        return weights, indices


class Gemma4Experts(nn.Module):
    """Sparse expert feed-forward, dispatched via grouped GEMM (mirrors qwen2_moe)."""

    def __init__(self, config: Gemma4Config):
        tc = config.text_config
        self.num_experts = tc.num_experts
        self.top_k = tc.top_k_experts
        self.hidden_size = tc.hidden_size
        moe_inter = tc.moe_intermediate_size
        self.e1_e3 = MixtralExperts(
            self.num_experts, in_features=tc.hidden_size, out_features=2 * moe_inter
        )
        self.e2 = MixtralExperts(
            self.num_experts, in_features=moe_inter, out_features=tc.hidden_size
        )

    def forward(self, x: Tensor, indices: Tensor, weights: Tensor):
        # x: [num_tokens, hidden_size]; indices/weights: [num_tokens, top_k]
        num_tokens = x.shape[0]

        def _expert_forward(xx: Tensor, indptr: Tensor):
            gate_up = self.e1_e3(xx, indptr)
            gate, up = op.split(gate_up, 2, axis=-1)
            return self.e2(op.gelu(gate, approximate="tanh") * up, indptr)

        if num_tokens == 1:
            out = _expert_forward(x, indices)
        else:
            cumsum = op_ext.moe_misc.moe_cumsum(indices, self.num_experts)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, indices)
            indptr = op_ext.moe_misc.get_indptr(
                cumsum, self.num_experts, num_tokens, inclusive=False, out_dtype="int32"
            )
            out = op.take(x, token_indices, axis=0)
            out = _expert_forward(out, indptr)
            out = op_ext.moe_misc.scatter_output(out, reverse_indices)
        out = out.reshape(num_tokens, self.top_k, self.hidden_size) * weights.reshape(
            num_tokens, self.top_k, 1
        )
        return op_ext.moe_misc.moe_sum(out, dim=1)


class Gemma4DecoderLayer(nn.Module):
    """A Gemma4 decoder layer: attention + (dense MLP || MoE)."""

    def __init__(self, config: Gemma4Config, layer_idx: int):
        tc = config.text_config
        eps = tc.rms_norm_eps
        h = tc.hidden_size
        self.hidden_size = h
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config)
        self.router = Gemma4Router(config)
        self.experts = Gemma4Experts(config)
        self.input_layernorm = nn.RMSNorm(h, -1, eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(h, -1, eps, bias=False)
        self.pre_feedforward_layernorm = nn.RMSNorm(h, -1, eps, bias=False)
        self.post_feedforward_layernorm = nn.RMSNorm(h, -1, eps, bias=False)
        self.post_feedforward_layernorm_1 = nn.RMSNorm(h, -1, eps, bias=False)
        self.post_feedforward_layernorm_2 = nn.RMSNorm(h, -1, eps, bias=False)
        self.pre_feedforward_layernorm_2 = nn.RMSNorm(h, -1, eps, bias=False)
        self.layer_scalar = nn.Parameter((1,))

    def forward(
        self, hidden_states: Tensor, positions: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ):
        b, s, _ = hidden_states.shape

        residual = hidden_states
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, positions, paged_kv_cache, layer_id)
        out = self.post_attention_layernorm(out)
        hidden_states = residual + out

        residual = hidden_states
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        h1 = self.post_feedforward_layernorm_1(mlp_out)

        residual_flat = op.reshape(residual, (b * s, self.hidden_size))
        weights, indices = self.router(residual_flat)
        experts_out = self.experts(self.pre_feedforward_layernorm_2(residual_flat), indices, weights)
        experts_out = op.reshape(experts_out, (b, s, self.hidden_size))
        h2 = self.post_feedforward_layernorm_2(experts_out)

        out = self.post_feedforward_layernorm(h1 + h2)
        hidden_states = residual + out
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4Config):
        tc = config.text_config
        assert tc.hidden_size % tc.num_attention_heads == 0
        self.hidden_size = tc.hidden_size
        self.embed_tokens = GemmaEmbedding("vocab_size", tc.hidden_size)
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(tc.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(tc.hidden_size, -1, tc.rms_norm_eps, bias=False)

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed * (self.hidden_size**0.5)
        positions = paged_kv_cache.get_query_positions(
            input_embed.shape[0] * input_embed.shape[1]
        )
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, positions, paged_kv_cache, layer_id)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        tc = config.text_config
        self.language_model = Gemma4TextModel(config)

        self.num_hidden_layers = tc.num_hidden_layers
        self.num_attention_heads = tc.num_attention_heads
        self.cache_kv_heads = tc.num_key_value_heads
        self.cache_head_dim = tc.global_head_dim
        self.layer_types = tc.layer_types
        self.rope_theta = tc.rope_theta_sliding  # unused by the cache (RopeMode.NONE)
        self.hidden_size = tc.hidden_size
        self.vocab_size = config.vocab_size
        self.final_logit_softcapping = tc.final_logit_softcapping
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def get_logits(self, hidden_states: Tensor):
        logits = self.language_model.embed_tokens.lm_head_forward(hidden_states)
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = op.tanh(logits / cap) * cap
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states = self.language_model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def embed(self, input_ids: Tensor):
        return self.language_model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.language_model(input_embed, paged_kv_cache)
        hidden_states = index_last_token(hidden_states)
        return self.get_logits(hidden_states), paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.language_model(input_embed, paged_kv_cache)
        return self.get_logits(hidden_states), paged_kv_cache

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        return self.batch_forward(input_embeds, paged_kv_cache, logit_positions), paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_forward(input_embeds, paged_kv_cache), paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        return self.batch_forward(input_embeds, paged_kv_cache), paged_kv_cache

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind=[
                "mha_sliding" if t == "sliding_attention" else "mha" for t in self.layer_types
            ],
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.cache_kv_heads,
            qk_head_dim=self.cache_head_dim,
            v_head_dim=self.cache_head_dim,
            rope_mode=RopeMode.NONE,  # RoPE is applied manually per layer
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
