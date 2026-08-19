"""Strict-online SemOpKT reference implementation used by all field variants."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.nn import functional as F

from semopkt.models.base import KTModel, KTOutput
from semopkt.models.layers import FeedForward
from semopkt.semantics.trainable import TrainableDescriptorEncoder


class FieldUpdateLayer(nn.Module):
    def __init__(self, dimensions: Mapping[str, int], architecture: Mapping[str, Any]):
        super().__init__()
        semantic = int(dimensions["semantic"])
        field = int(dimensions["field"])
        context = int(dimensions["context"])
        attention = int(dimensions["attention"])
        response = int(dimensions["response"])
        hidden = int(dimensions["hidden"])
        rank = int(architecture["rank"])
        dropout = float(architecture["dropout"])
        self.field_size = field
        self.rank = rank
        self.attention_size = attention
        self.response_conditioning = bool(architecture.get("response_conditioning", True))
        self.update_mode = str(architecture.get("update_mode", "semopkt"))
        self.response_embedding = nn.Embedding(2, response)
        self.context = FeedForward(
            field + semantic + response, context, [hidden, hidden], dropout
        )
        self.query = nn.Linear(context, rank * attention)
        self.key = nn.Linear(semantic + field, rank * attention)
        self.response_value = nn.Linear(context, rank * field)
        self.semantic_bias = FeedForward(4 * semantic, rank, [hidden, hidden], dropout)
        gate_input = field + semantic + context + field
        self.gate = FeedForward(gate_input, field, [hidden, hidden], dropout)
        self.value = FeedForward(field, field, [hidden], dropout)
        self.gru = nn.GRUCell(context + semantic, field)
        self.anchor_attention = nn.MultiheadAttention(
            embed_dim=field,
            num_heads=max(1, min(8, field // 8)),
            dropout=dropout,
            batch_first=True,
        )

    def _pair_features(self, coordinate: Tensor, anchors: Tensor, batch: int) -> Tensor:
        target = coordinate[:, None, :].expand(batch, anchors.shape[0], -1)
        anchor = anchors[None, :, :].expand(batch, -1, -1)
        return torch.cat([target, anchor, target - anchor, target * anchor], dim=-1)

    def forward(
        self, state: Tensor, coordinate: Tensor, field_value: Tensor, response: Tensor, anchors: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, inducing, _ = state.shape
        response_features = self.response_embedding(response.long().clamp(0, 1))
        if not self.response_conditioning:
            response_features = torch.zeros_like(response_features)
        context = self.context(torch.cat([field_value, coordinate, response_features], dim=-1))
        if self.update_mode == "gru":
            inputs = torch.cat(
                [
                    context[:, None, :].expand(batch, inducing, -1),
                    anchors[None, :, :].expand(batch, -1, -1),
                ],
                dim=-1,
            )
            updated = self.gru(inputs.reshape(batch * inducing, -1), state.reshape(batch * inducing, -1))
            increment = updated.reshape(batch, inducing, -1) - state
            return updated, {"proposal": increment, "gate": torch.ones_like(increment)}
        if self.update_mode == "transformer":
            query = state
            context_token = field_value[:, None, :]
            attended, _ = self.anchor_attention(query, context_token, context_token, need_weights=False)
            increment = attended
            return state + increment, {"proposal": increment, "gate": torch.ones_like(increment)}
        queries = self.query(context).reshape(batch, self.rank, self.attention_size)
        key_inputs = torch.cat(
            [anchors[None, :, :].expand(batch, -1, -1), state], dim=-1
        )
        keys = self.key(key_inputs).reshape(
            batch, inducing, self.rank, self.attention_size
        )
        values = self.response_value(context).reshape(batch, self.rank, self.field_size)
        pair_features = self._pair_features(coordinate, anchors, batch)
        bias = self.semantic_bias(pair_features)
        scores = torch.einsum("brh,bmrh->bmr", queries, keys) / math.sqrt(self.attention_size)
        scores = scores + bias
        if self.update_mode == "fixed_rbf":
            similarity = torch.einsum("bd,md->bm", coordinate, anchors).clamp(-1.0, 1.0)
            scores = similarity[:, :, None].expand(-1, -1, self.rank) / 0.1
        weights = torch.softmax(scores.float(), dim=1).to(state.dtype)
        if self.update_mode == "pointwise":
            winner = weights.argmax(dim=1, keepdim=True)
            hard = torch.zeros_like(weights).scatter_(1, winner, 1.0)
            weights = hard + weights - weights.detach()
        proposal = torch.einsum("bmr,brd->bmd", weights, values)
        if self.update_mode == "gkt":
            adjacency = F.relu(torch.einsum("md,nd->mn", anchors, anchors))
            adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            proposal = torch.einsum("mn,bnd->bmd", adjacency, proposal)
        gate_input = torch.cat(
            [
                state,
                anchors[None, :, :].expand(batch, -1, -1),
                context[:, None, :].expand(batch, inducing, -1),
                proposal,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        if self.update_mode == "linear":
            increment = proposal
            gate = torch.ones_like(proposal)
        else:
            increment = gate * self.value(proposal)
        return state + increment, {"proposal": proposal, "gate": gate, "weights": weights}


class SemOpKT(KTModel):
    model_name = "SemOpKT"
    uses_semantic_descriptors = True

    def __init__(
        self,
        config: Mapping[str, Any],
        concept_embeddings: np.ndarray | Tensor,
        concept_texts: Sequence[str] | None = None,
    ):
        super().__init__()
        self.config = dict(config)
        dimensions = dict(config["dimensions"])
        architecture = dict(config["architecture"])
        regularization = dict(config["regularization"])
        text_size = int(dimensions["text"])
        semantic = int(dimensions["semantic"])
        field = int(dimensions["field"])
        hidden = int(dimensions["hidden"])
        inducing = int(architecture["inducing_points"])
        dropout = float(architecture["dropout"])
        embeddings = torch.as_tensor(concept_embeddings, dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != text_size:
            raise ValueError(
                f"Concept embeddings have shape {tuple(embeddings.shape)}, expected (*, {text_size})"
            )
        trainable_concepts = bool(
            architecture.get("trainable_concept_embeddings", False)
        )
        if trainable_concepts:
            self.concept_embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer("concept_embeddings", embeddings, persistent=True)
        self.uses_semantic_descriptors = not trainable_concepts
        finetuning = str(config.get("text_encoder", {}).get("finetuning", "frozen"))
        self.descriptor_encoder: TrainableDescriptorEncoder | None = None
        if finetuning in {"partial", "full"}:
            if trainable_concepts:
                raise ValueError(
                    "Trainable identifier coordinates and encoder fine-tuning are mutually exclusive"
                )
            if concept_texts is None or len(concept_texts) != len(embeddings):
                raise ValueError("Encoder fine-tuning requires one descriptor for every concept")
            self.descriptor_encoder = TrainableDescriptorEncoder(
                concept_texts, config["text_encoder"]
            )
            if self.descriptor_encoder.output_dimension != text_size:
                raise ValueError(
                    "Configured text dimension does not match differentiable encoder output"
                )
        self.semantic_projection = nn.Linear(text_size, semantic)
        self.interpolation = FeedForward(4 * semantic, 1, [hidden, hidden], dropout)
        temperature = float(architecture.get("interpolation_temperature", 1.0))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature), dtype=torch.float32),
            requires_grad=bool(architecture.get("learnable_temperature", True)),
        )
        self.anchors = nn.Parameter(torch.randn(inducing, semantic))
        self.population_prior = nn.Parameter(torch.zeros(inducing, field))
        if architecture.get("population_prior_initialization") == "normal":
            nn.init.normal_(self.population_prior, std=0.02)
        layer_count = int(architecture["layers"])
        if bool(architecture.get("layer_sharing", False)):
            shared = FieldUpdateLayer(dimensions, architecture)
            self.update_layers = nn.ModuleList([shared for _ in range(layer_count)])
        else:
            self.update_layers = nn.ModuleList(
                [FieldUpdateLayer(dimensions, architecture) for _ in range(layer_count)]
            )
        self.direct_coordinate_path = bool(architecture.get("direct_coordinate_path", True))
        prediction_input = field + semantic if self.direct_coordinate_path else field
        self.prediction_head = FeedForward(prediction_input, 1, [hidden, hidden], dropout)
        self.epsilon = float(architecture.get("epsilon", 1.0e-8))
        self.smoothness_coefficient = float(regularization.get("smoothness", 0.0))
        self.stability_coefficient = float(regularization.get("stability", 0.0))
        self.normalize_trainable_anchors = bool(
            architecture.get("normalize_anchors_after_step", True)
        )
        self.register_buffer("graph_edges", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("graph_weights", torch.empty((0,), dtype=torch.float32))
        self._regularization_concepts: tuple[int, ...] | None = None
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.initialize_anchors()

    def concept_coordinates(self) -> Tensor:
        embeddings = (
            self.descriptor_encoder()
            if self.descriptor_encoder is not None
            else self.concept_embeddings
        )
        projected = self.semantic_projection(embeddings)
        return F.normalize(projected, dim=-1, eps=self.epsilon)

    @torch.no_grad()
    def initialize_anchors(
        self, concept_indices: Sequence[int] | None = None
    ) -> None:
        coordinates = self.concept_coordinates().detach().cpu().numpy()
        if concept_indices is not None:
            selected = np.asarray(
                sorted(set(int(index) for index in concept_indices)), dtype=np.int64
            )
            if len(selected) == 0:
                raise ValueError("Anchor initialization requires a training concept")
            coordinates = coordinates[selected]
        inducing = self.anchors.shape[0]
        if len(coordinates) >= inducing:
            model = KMeans(n_clusters=inducing, init="k-means++", n_init=10, random_state=0)
            initial = model.fit(coordinates).cluster_centers_
        else:
            repeat = int(math.ceil(inducing / max(1, len(coordinates))))
            initial = np.tile(coordinates, (repeat, 1))[:inducing]
        initial = initial / np.clip(np.linalg.norm(initial, axis=1, keepdims=True), 1.0e-8, None)
        self.anchors.copy_(torch.as_tensor(initial, dtype=self.anchors.dtype, device=self.anchors.device))

    @torch.no_grad()
    def build_smoothness_graph(self) -> None:
        settings = self.config["regularization"]
        neighbors = int(settings.get("graph_neighbors", 5))
        temperature = float(settings.get("graph_rbf_temperature", 0.1))
        all_frozen = F.normalize(self.concept_embeddings.float(), dim=-1)
        selected = (
            torch.as_tensor(
                self._regularization_concepts,
                dtype=torch.long,
                device=all_frozen.device,
            )
            if self._regularization_concepts is not None
            else torch.arange(len(all_frozen), device=all_frozen.device)
        )
        frozen = all_frozen[selected]
        distance = 1.0 - frozen @ frozen.T
        count = frozen.shape[0]
        if count < 2:
            self.graph_edges = torch.empty((2, 0), dtype=torch.long, device=frozen.device)
            self.graph_weights = torch.empty((0,), dtype=torch.float32, device=frozen.device)
            return
        k = min(neighbors, count - 1)
        nearest = torch.topk(distance, k=k + 1, largest=False).indices[:, 1:]
        edge_set: set[tuple[int, int]] = set()
        for source in range(count):
            for target in nearest[source].tolist():
                edge_set.add((min(source, int(target)), max(source, int(target))))
        edges = sorted(edge_set)
        local_edges = torch.tensor(edges, dtype=torch.long, device=frozen.device).T
        edge_tensor = selected[local_edges]
        edge_distance = distance[local_edges[0], local_edges[1]]
        weights = torch.exp(-edge_distance / max(temperature, 1.0e-8))
        self.graph_edges = edge_tensor
        self.graph_weights = weights

    def configure_training_concepts(self, concept_indices: Sequence[int]) -> None:
        values = tuple(sorted(set(int(index) for index in concept_indices)))
        if not values:
            raise ValueError("Smoothness graph requires at least one training concept")
        self._regularization_concepts = values
        self.initialize_anchors(values)

    def interpolation_weights(self, coordinates: Tensor) -> Tensor:
        anchors = F.normalize(self.anchors, dim=-1, eps=self.epsilon)
        if coordinates.ndim == 2:
            target = coordinates[:, None, :].expand(-1, anchors.shape[0], -1)
            anchor = anchors[None, :, :].expand(coordinates.shape[0], -1, -1)
        elif coordinates.ndim == 3:
            target = coordinates[:, :, None, :].expand(-1, -1, anchors.shape[0], -1)
            anchor = anchors[None, None, :, :].expand(
                coordinates.shape[0], coordinates.shape[1], -1, -1
            )
        else:
            raise ValueError("Coordinates must have shape [B,D] or [B,Q,D]")
        features = torch.cat([target, anchor, target - anchor, target * anchor], dim=-1)
        logits = self.interpolation(features).squeeze(-1)
        temperature = self.log_temperature.exp().clamp(0.02, 20.0)
        return torch.softmax((logits / temperature).float(), dim=-1).to(coordinates.dtype)

    def field_query(self, state: Tensor, coordinates: Tensor) -> Tensor:
        weights = self.interpolation_weights(coordinates)
        if coordinates.ndim == 2:
            return torch.einsum("bm,bmd->bd", weights, state)
        return torch.einsum("bqm,bmd->bqd", weights, state)

    def _smoothness(self, state: Tensor, coordinates: Tensor, active: Tensor) -> Tensor:
        if (
            self.smoothness_coefficient <= 0
            or self.graph_edges.numel() == 0
            or not torch.any(active)
        ):
            return state.new_zeros(())
        selected_state = state[active]
        sources = coordinates[self.graph_edges[0]]
        targets = coordinates[self.graph_edges[1]]
        source_values = self.field_query(
            selected_state, sources[None, :, :].expand(selected_state.shape[0], -1, -1)
        )
        target_values = self.field_query(
            selected_state, targets[None, :, :].expand(selected_state.shape[0], -1, -1)
        )
        squared = (source_values - target_values).square().sum(dim=-1)
        return (squared * self.graph_weights[None, :]).mean()

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        concept_indices = batch["concept_indices"].long()
        labels = batch["labels"].float()
        valid_mask = batch["valid_mask"].bool()
        batch_size, length = concept_indices.shape
        all_coordinates = self.concept_coordinates()
        anchors = F.normalize(self.anchors, dim=-1, eps=self.epsilon)
        state = self.population_prior[None, :, :].expand(batch_size, -1, -1).clone()
        logits: list[Tensor] = []
        stability_terms: list[Tensor] = []
        smoothness_terms: list[Tensor] = []
        last_weights: Tensor | None = None
        for step in range(length):
            active = valid_mask[:, step]
            coordinate = all_coordinates[concept_indices[:, step]]
            field_value = self.field_query(state, coordinate)
            prediction_input = (
                torch.cat([field_value, coordinate], dim=-1)
                if self.direct_coordinate_path
                else field_value
            )
            # The current label is first accessed below, after this logit has been emitted.
            logits.append(self.prediction_head(prediction_input).squeeze(-1))
            if self.smoothness_coefficient > 0:
                smoothness_terms.append(self._smoothness(state, all_coordinates, active))
            updated = state
            layer_auxiliary: dict[str, Tensor] = {}
            for layer in self.update_layers:
                current_field = self.field_query(updated, coordinate)
                updated, layer_auxiliary = layer(
                    updated, coordinate, current_field, labels[:, step], anchors
                )
            active_state = active[:, None, None]
            delta = updated - state
            stability_terms.append(delta.square()[active_state.expand_as(delta)].mean() if torch.any(active) else state.new_zeros(()))
            state = torch.where(active_state, updated, state)
            last_weights = layer_auxiliary.get("weights")
        stacked = torch.stack(logits, dim=1)
        smoothness = (
            torch.stack(smoothness_terms).mean() * self.smoothness_coefficient
            if smoothness_terms
            else stacked.new_zeros(())
        )
        stability = (
            torch.stack(stability_terms).mean() * self.stability_coefficient
            if stability_terms
            else stacked.new_zeros(())
        )
        auxiliary: dict[str, Any] = {"final_state": state}
        if last_weights is not None:
            auxiliary["last_update_weights"] = last_weights
        return KTOutput(stacked, valid_mask, smoothness, stability, auxiliary)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        if self.normalize_trainable_anchors:
            self.anchors.copy_(F.normalize(self.anchors, dim=-1, eps=self.epsilon))
        self.log_temperature.clamp_(math.log(0.02), math.log(20.0))
