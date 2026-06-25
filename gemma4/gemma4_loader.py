"""
This file specifies how MLC's Gemma4 parameters map from HuggingFace formats (PyTorch /
safetensors).  Only the text decoder is mapped; vision / audio tower weights are ignored.

Unlike Gemma1/2/3, Gemma4's RMSNorm does NOT add 1 to its weight, so the weights are copied
verbatim (no ``+1`` fusion).  In the multimodal checkpoint the text weights live under the
``model.language_model.`` prefix; the experts are stored as pre-stacked 3D tensors
(``experts.gate_up_proj`` / ``experts.down_proj``) and the LM head is tied to the embedding.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .gemma4_model import Gemma4Config, Gemma4ForCausalLM


def huggingface(model_config: Gemma4Config, quantization: Quantization) -> ExternMapping:
    """Create the HuggingFace -> MLC parameter mapping for Gemma4."""
    model = Gemma4ForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    is_text_model = model_config.is_text_model

    def name_transform(mlc_name: str) -> str:
        # MLC names are rooted at ``language_model.`` (Gemma4ForCausalLM.language_model).
        suffix = mlc_name[len("language_model.") :]
        # Experts are stored fused in the checkpoint.
        suffix = suffix.replace(".experts.e1_e3.weight", ".experts.gate_up_proj")
        suffix = suffix.replace(".experts.e2.weight", ".experts.down_proj")
        if is_text_model:
            return f"model.{suffix}"
        return f"model.language_model.{suffix}"

    mapping = ExternMapping()
    for mlc_name, mlc_param in named_parameters.items():
        mapping.add_mapping(
            mlc_name,
            [name_transform(mlc_name)],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
    return mapping
