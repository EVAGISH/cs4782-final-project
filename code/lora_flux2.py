import torch.nn as nn

from lora import (
    LoRALinear,
    get_lora_state_dict,
    get_lora_parameter_names,
    summarize_lora_state_dict,
)


__all__ = [
    "patch_flux2_transformer_with_lora",
    "get_lora_state_dict",
    "get_lora_parameter_names",
    "summarize_lora_state_dict",
]


def _wrap_attr(module: nn.Module, attr: str, rank: int, alpha: float, lora_params: list):
    original = getattr(module, attr)
    if not isinstance(original, nn.Linear):
        return
    wrapped = LoRALinear(original, rank, alpha)
    setattr(module, attr, wrapped)
    lora_params.extend([wrapped.lora_A, wrapped.lora_B])


def _wrap_index(container, index: int, rank: int, alpha: float, lora_params: list):
    original = container[index]
    if not isinstance(original, nn.Linear):
        return
    wrapped = LoRALinear(original, rank, alpha)
    container[index] = wrapped
    lora_params.extend([wrapped.lora_A, wrapped.lora_B])


def patch_flux2_transformer_with_lora(transformer, rank: int, alpha: float, target_dual_text: bool = True):
    # Lazy import: lets the module load even if diffusers lacks Flux2 (older versions).
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Attention,
        Flux2ParallelSelfAttention,
    )

    lora_params: list = []
    for module in transformer.modules():
        if isinstance(module, Flux2Attention):
            for sub in ("to_q", "to_k", "to_v"):
                _wrap_attr(module, sub, rank, alpha, lora_params)
            # to_out is nn.ModuleList([Linear, Dropout]); also handle Sequential for forward compat.
            to_out = getattr(module, "to_out", None)
            if isinstance(to_out, (nn.ModuleList, nn.Sequential)) and len(to_out) > 0:
                _wrap_index(to_out, 0, rank, alpha, lora_params)
            if target_dual_text:
                for sub in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
                    if hasattr(module, sub):
                        _wrap_attr(module, sub, rank, alpha, lora_params)
        elif isinstance(module, Flux2ParallelSelfAttention):
            _wrap_attr(module, "to_qkv_mlp_proj", rank, alpha, lora_params)
            _wrap_attr(module, "to_out", rank, alpha, lora_params)

    return lora_params
