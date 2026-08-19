"""Differentiable descriptor encoding for partial and full encoder fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor, nn
from torch.nn import functional as F

from semopkt.data.schema import normalize_text


def _encoder_blocks(model: nn.Module) -> list[nn.Module]:
    candidates = (
        ("encoder", "layer"),
        ("transformer", "layer"),
        ("encoder", "layers"),
        ("transformer", "layers"),
    )
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        blocks = getattr(parent, child_name, None) if parent is not None else None
        if isinstance(blocks, (nn.ModuleList, list, tuple)):
            return list(blocks)
    base_prefix = getattr(model, "base_model_prefix", "")
    base = getattr(model, base_prefix, None) if base_prefix else None
    if isinstance(base, nn.Module) and base is not model:
        return _encoder_blocks(base)
    return []


class TrainableDescriptorEncoder(nn.Module):
    """Pinned transformer encoder with fixed, normalized descriptor tokens."""

    def __init__(self, texts: Sequence[str], config: Mapping[str, Any]):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("transformers is required for encoder fine-tuning") from error
        mode = str(config.get("finetuning", "frozen"))
        if mode not in {"partial", "full"}:
            raise ValueError(
                f"Differentiable descriptor encoder requires partial/full mode, got {mode}"
            )
        model_id = str(config["model_id"])
        revision = str(config["revision"])
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        prefix = str(config.get("instruction_prefix", ""))
        normalized = [prefix + normalize_text(text) for text in texts]
        tokens = tokenizer(
            normalized,
            padding=True,
            truncation=True,
            max_length=int(config.get("max_length", 128)),
            return_tensors="pt",
        )
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.register_buffer("input_ids", tokens["input_ids"].long(), persistent=True)
        self.register_buffer(
            "attention_mask", tokens["attention_mask"].long(), persistent=True
        )
        if "token_type_ids" in tokens:
            self.register_buffer(
                "token_type_ids", tokens["token_type_ids"].long(), persistent=True
            )
        else:
            self.token_type_ids = None
        self.normalize_output = bool(config.get("normalize", True))
        for parameter in self.model.parameters():
            parameter.requires_grad_(mode == "full")
        if mode == "partial":
            blocks = _encoder_blocks(self.model)
            unfrozen = int(config.get("partial_unfrozen_layers", 2))
            if not blocks or unfrozen < 1:
                raise ValueError("Partial fine-tuning could not resolve encoder blocks")
            for block in blocks[-min(unfrozen, len(blocks)) :]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Encoder configuration does not expose hidden_size")
        self.output_dimension = int(hidden_size)

    def forward(self) -> Tensor:
        arguments: dict[str, Tensor] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if isinstance(self.token_type_ids, Tensor):
            arguments["token_type_ids"] = self.token_type_ids
        output = self.model(**arguments)
        hidden = output.last_hidden_state
        mask = self.attention_mask[:, :, None].to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1) if self.normalize_output else pooled

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
