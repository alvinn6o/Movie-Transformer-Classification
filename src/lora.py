"""Low-Rank Adaptation (Hu et al., 2021), written out instead of imported.

A frozen Linear W gets a trainable low-rank update:
    y = W x + (alpha / r) * B A x,    A: d_in -> r,  B: r -> d_out,  B initialised to 0
so training starts exactly at the pretrained model and only r * (d_in + d_out)
parameters per adapted layer are learned.
"""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(model: nn.Module, target_suffixes=("q_proj", "v_proj", "q_lin", "v_lin"),
               r: int = 8, alpha: int = 16, dropout: float = 0.05) -> list[str]:
    """Swap every nn.Linear whose name ends with a target suffix for a LoRALinear."""
    targets = [name for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear) and name.split(".")[-1] in target_suffixes]
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return targets


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
