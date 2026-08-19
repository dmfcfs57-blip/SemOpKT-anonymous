"""Reusable neural layers for field interpolation and sequential baselines."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int],
        dropout: float,
        final_activation: bool = False,
    ):
        super().__init__()
        sizes = [input_size, *hidden_sizes, output_size]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            layers.append(nn.Linear(source, target))
            is_final = index == len(sizes) - 2
            if not is_final or final_activation:
                layers.extend([nn.LayerNorm(target), nn.SiLU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


def masked_mean(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.sum(dim=dimension).clamp_min(1.0)
    return (values * weights).sum(dim=dimension) / denominator

