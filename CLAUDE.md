# CLAUDE.md — Adding New Model Support in MLC-LLM

## Project Overview

MLC-LLM is a universal LLM deployment engine that compiles models to run natively on various hardware backends (CUDA, Metal, Vulkan, WebGPU, etc.) via Apache TVM. Adding a new model means:

1. Registering the model architecture in Python (forward pass, attention, FFN blocks)
2. Registering config/tokenizer mappings
3. Adding quantization support
4. Wiring up the conversation template
5. Running compilation + smoke tests

---

## Repository Layout (relevant paths)

```
mlc-llm/
├── python/mlc_llm/
│   ├── model/                   # ← Model architecture definitions
│   │   ├── model.py             #   Registry: MODEL_PRESETS dict
│   │   ├── llama/               #   Reference implementation
│   │   │   ├── llama_model.py
│   │   │   ├── llama_quantization.py
│   │   │   └── __init__.py
│   │   └── <your_model>/        # ← Create this directory
│   ├── conversation_template/
│   │   └── registry.py          # Chat template registration
│   ├── compiler/
│   │   └── compile.py           # Compilation entry point
│   └── interface/
│       └── gen_config.py        # mlc-chat-config.json generation
├── cpp/                         # C++ runtime (rarely touched for new models)
├── tests/
│   └── python/
│       └── model/               # Unit tests per model
└── mlc-package-config.json      # Example package configs
```

---

## Step-by-Step: Adding a New Model

### 1. Create the Model Directory

```
python/mlc_llm/model/<your_model>/
├── __init__.py
├── <your_model>_model.py        # Architecture (nn.Module subclasses)
└── <your_model>_quantization.py # Quantization spec (weight mapping)
```

### 2. Implement the Architecture (`<your_model>_model.py`)

Follow the pattern from `llama/llama_model.py`. Key requirements:

```python
import dataclasses
from typing import Optional
from tvm import relax
from tvm.relax.frontend import nn
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support.config import ConfigBase

@dataclasses.dataclass
class YourModelConfig(ConfigBase):
    """Maps to HuggingFace config.json fields."""
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int          # for GQA; set == num_attention_heads if MHA
    head_dim: int                     # hidden_size // num_attention_heads usually
    vocab_size: int
    context_window_size: int = 4096
    prefill_chunk_size: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    # quantization fields (filled by mlc_llm internals)
    kwargs: dict = dataclasses.field(default_factory=dict)


class YourModelAttention(nn.Module):
    def __init__(self, config: YourModelConfig):
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

    def forward(self, hidden_states: relax.Expr, paged_kv_cache: PagedKVCache, layer_id: int):
        # project, reshape to (batch, seq, heads, head_dim)
        q = nn.reshape(self.q_proj(hidden_states), (..., self.num_q_heads, self.head_dim))
        k = nn.reshape(self.k_proj(hidden_states), (..., self.num_kv_heads, self.head_dim))
        v = nn.reshape(self.v_proj(hidden_states), (..., self.num_kv_heads, self.head_dim))
        # RoPE + attention via PagedKVCache
        output = paged_kv_cache.attention_with_fused_qkv(layer_id, nn.concat([q, k, v], axis=-2), self.num_q_heads)
        return self.o_proj(nn.reshape(output, (..., self.num_q_heads * self.head_dim)))


class YourModelMLP(nn.Module):
    def __init__(self, config: YourModelConfig):
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))  # SwiGLU


class YourModelDecoderLayer(nn.Module):
    def __init__(self, config: YourModelConfig):
        self.self_attn = YourModelAttention(config)
        self.mlp = YourModelMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, 1e-5, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, 1e-5, bias=False)

    def forward(self, hidden_states, paged_kv_cache, layer_id):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), paged_kv_cache, layer_id)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class YourModel(nn.Module):
    def __init__(self, config: YourModelConfig):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([YourModelDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, -1, 1e-5, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # RoPE mode: NORMAL, INLINE, or NONE
        self.rope_mode = RopeMode.NORMAL
        self.config = config

    def to(self, spec):
        """Apply quantization spec."""
        self.embed_tokens.to(spec=spec)
        for layer in self.layers:
            layer.to(spec=spec)
        self.lm_head.to(spec=spec)
        return self

    def embed(self, input_ids):
        return self.embed_tokens(input_ids)

    def prefill(self, input_embed, paged_kv_cache):
        hidden = input_embed
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, paged_kv_cache, i)
        hidden = self.norm(hidden)
        return self.lm_head(hidden), paged_kv_cache

    def decode(self, input_embed, paged_kv_cache):
        return self.prefill(input_embed, paged_kv_cache)  # same graph, different shape

    def softmax_with_temperature(self, logits, temperature):
        return nn.softmax(logits / temperature, axis=-1)

    def create_paged_kv_cache(self, ...):
        # delegate to mlc_llm.nn.kv_cache helpers
        ...
```

**Checklist for architecture:**
- [ ] `ConfigBase` subclass with all HF `config.json` fields
- [ ] Correct `num_key_value_heads` for GQA/MQA
- [ ] `RopeMode` set appropriately (`NORMAL` for standard RoPE, `INLINE` if fused into QK, `NONE` for ALiBi/NoPE)
- [ ] `prefill` and `decode` entry points
- [ ] `to(spec)` for quantization

---

### 3. Implement Quantization Spec (`<your_model>_quantization.py`)

```python
from mlc_llm.quantization import quantization_schemes
from mlc_llm.quantization.quantization import QuantizationSpec, GroupQuantizationSpec

def get_param_quant_kind(name: str, param_info) -> QuantizationSpec:
    """Map parameter names to their quantization treatment."""
    if "embed_tokens" in name or "lm_head" in name:
        return quantization_schemes.EMBEDDING_SPEC
    if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]):
        return quantization_schemes.WEIGHT_ONLY_SPEC
    # norms, biases → keep as fp
    return quantization_schemes.NO_QUANT
```

Reuse existing schemes (`q4f16_1`, `q4f32_1`, `q0f16`, etc.) — only write a custom spec if the weight layout is non-standard (e.g., fused QKV, MoE gates).

---

### 4. Register the Model

In `python/mlc_llm/model/model.py`, add to `MODEL_PRESETS`:

```python
from .your_model import YourModelConfig, YourModel, get_param_quant_kind

MODEL_PRESETS: Dict[str, ModelInfo] = {
    # ... existing entries ...
    "your_model": ModelInfo(
        name="your_model",
        config=YourModelConfig,
        model=YourModel,
        quantization_scheme=get_param_quant_kind,
        hf_config_name="YourModelForCausalLM",   # matches architectures[] in HF config.json
    ),
}
```

`hf_config_name` must match the string in the HuggingFace `config.json` under `"architectures"`.

---

### 5. Add Conversation Template

In `python/mlc_llm/conversation_template/registry.py`:

```python
ConvTemplate(
    name="your_model",
    system_template=f"{system_message}",
    system_message="You are a helpful assistant.",
    system_prefix_token_ids=None,     # or BOS id
    roles={"user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool|>"},
    role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
    messages=[],
    seps=["<|end|>\n", "<|end|>\n"],
    role_content_sep="",
    role_empty_sep="",
    stop_str=["<|end|>", "<|endoftext|>"],
    stop_token_ids=[32000, 32007],    # look up in tokenizer
    add_role_after_system_message=True,
)
```

Register it by adding to `CONV_TEMPLATES` dict in the same file.

---

### 6. Update `gen_config.py` (if needed)

If your model uses a non-standard tokenizer class or has special generation defaults, add a branch in `python/mlc_llm/interface/gen_config.py`:

```python
if model_config.model_type == "your_model":
    chat_config.temperature = 0.7
    chat_config.top_p = 0.9
```

---

### 7. Compile and Convert Weights

```bash
# Convert HuggingFace weights → MLC format
mlc_llm convert_weight ./your-model-hf/ \
    --quantization q4f16_1 \
    -o ./your-model-q4f16_1-MLC/

# Generate mlc-chat-config.json
mlc_llm gen_config ./your-model-hf/ \
    --quantization q4f16_1 \
    --conv-template your_model \
    -o ./your-model-q4f16_1-MLC/

# Compile to .so for target device
mlc_llm compile ./your-model-q4f16_1-MLC/mlc-chat-config.json \
    --device cuda \
    -o ./your-model-q4f16_1-MLC/lib.so
```

---

### 8. Write Tests

Add `tests/python/model/test_your_model.py`:

```python
import pytest
from mlc_llm.model import MODEL_PRESETS
from mlc_llm.model.your_model import YourModelConfig, YourModel

def test_config_roundtrip():
    cfg = YourModelConfig(hidden_size=2048, intermediate_size=5632,
                           num_hidden_layers=24, num_attention_heads=16,
                           num_key_value_heads=8, head_dim=128,
                           vocab_size=32000)
    assert cfg.head_dim == 128

def test_model_registered():
    assert "your_model" in MODEL_PRESETS

def test_forward_shape(tmp_path):
    """Smoke test: compile a tiny config and check output rank."""
    ...
```

Run tests:
```bash
pytest tests/python/model/test_your_model.py -v
```

---

## Common Pitfalls

| Issue | Fix |
|---|---|
| `hf_config_name` mismatch | Check `architectures` field in model's `config.json` exactly |
| Wrong `head_dim` | Some models (e.g. Gemma) set `head_dim` independently; don't assume `hidden_size // num_heads` |
| FlashAttention-2 incompatibility | Non-standard Q/K head dims (e.g. Gemma 4's dual head dims) break FA2 on pre-Blackwell hardware; set `rope_mode=RopeMode.INLINE` or use fallback attention |
| Quant spec missing layers | Run `model.named_parameters()` and verify every weight hits a non-default spec |
| Conversation template stop tokens wrong | Always verify `stop_token_ids` from the tokenizer's `special_tokens_map.json` |
| MoE models | See `mixtral/` for MoE routing reference; expert weight sharding needs a custom quant spec |

---

## Reference Implementations to Study

| Model type | Directory |
|---|---|
| Dense decoder (LLaMA-style) | `python/mlc_llm/model/llama/` |
| GQA variant | `python/mlc_llm/model/mistral/` |
| MoE | `python/mlc_llm/model/mixtral/` |
| Phi-style tied embeddings | `python/mlc_llm/model/phi/` |
| Vision-language | `python/mlc_llm/model/llava/` |

---

## Quick Reference: Key MLC-LLM Abstractions

- **`PagedKVCache`** — manages KV cache across requests; always passed into attention layers, never stored on the module
- **`RopeMode`** — `NORMAL`: RoPE applied inside `PagedKVCache`; `INLINE`: apply RoPE manually before passing QK; `NONE`: no positional encoding
- **`nn.Module`** — MLC's relax-based module, mirrors PyTorch API but traces to TVM Relax IR
- **`QuantizationSpec`** — maps a parameter to a quantization treatment; returned per-param from `get_param_quant_kind`
- **`ConfigBase`** — base dataclass with HF config loading helpers; unknown keys go into `kwargs`