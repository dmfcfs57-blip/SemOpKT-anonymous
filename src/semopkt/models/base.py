"""Common model output and loss contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class KTOutput:
    logits: Tensor
    valid_mask: Tensor
    smoothness_loss: Tensor
    stability_loss: Tensor
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.logits)


class KTModel(nn.Module):
    model_name = "KTModel"
    uses_semantic_descriptors = False

    def loss(self, output: KTOutput, labels: Tensor) -> tuple[Tensor, dict[str, float]]:
        selected = output.valid_mask.bool()
        if not torch.any(selected):
            raise ValueError("Batch has no valid target")
        prediction_loss = F.binary_cross_entropy_with_logits(
            output.logits[selected].float(), labels[selected].float()
        )
        total = prediction_loss + output.smoothness_loss + output.stability_loss
        parts = {
            "prediction": float(prediction_loss.detach().cpu()),
            "smoothness": float(output.smoothness_loss.detach().cpu()),
            "stability": float(output.stability_loss.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total, parts

    def normalize_parameters(self) -> None:
        """Apply constrained parameter projections after an optimizer step."""

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def zero_regularizer(reference: Tensor) -> Tensor:
    return reference.new_zeros(())
