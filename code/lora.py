import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x):
        base = self.original(x)
        x_lora = x.to(self.lora_A.dtype)
        lora = F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling
        return base + lora.to(base.dtype)


def patch_unet_with_lora(unet, rank: int, alpha: float):
    lora_params = []
    for module in unet.modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
            for sub_name in ("to_q", "to_k", "to_v"):
                original = getattr(module, sub_name)
                if isinstance(original, nn.Linear):
                    wrapped = LoRALinear(original, rank, alpha)
                    setattr(module, sub_name, wrapped)
                    lora_params.extend([wrapped.lora_A, wrapped.lora_B])
            to_out = getattr(module, "to_out", None)
            if isinstance(to_out, nn.Sequential) and len(to_out) > 0 and isinstance(to_out[0], nn.Linear):
                wrapped = LoRALinear(to_out[0], rank, alpha)
                to_out[0] = wrapped
                lora_params.extend([wrapped.lora_A, wrapped.lora_B])
    return lora_params


def get_lora_state_dict(unet):
    return {n: p.detach().cpu() for n, p in unet.named_parameters() if "lora_" in n}


def get_lora_parameter_names(unet) -> set[str]:
    return {n for n, _ in unet.named_parameters() if "lora_" in n}


def summarize_lora_state_dict(state: dict[str, torch.Tensor]) -> dict[str, float | int | str]:
    digest = hashlib.sha256()
    total_abs = 0.0
    total_numel = 0
    max_abs = 0.0

    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        tensor_abs = tensor.abs()
        total_abs += tensor_abs.sum().item()
        total_numel += tensor.numel()
        max_abs = max(max_abs, tensor_abs.max().item())

    return {
        "num_tensors": len(state),
        "mean_abs": total_abs / total_numel if total_numel else 0.0,
        "max_abs": max_abs,
        "sha256": digest.hexdigest()[:12],
    }
