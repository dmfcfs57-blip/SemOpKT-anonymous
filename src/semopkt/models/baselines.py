"""Clean-room fair-input implementations of the manuscript baseline families."""

from __future__ import annotations

import math
import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from semopkt.models.base import KTModel, KTOutput, zero_regularizer
from semopkt.models.layers import FeedForward, causal_mask


def _active_select(previous: Tensor, updated: Tensor, active: Tensor) -> Tensor:
    shape = [active.shape[0]] + [1] * (previous.ndim - 1)
    return torch.where(active.reshape(shape), updated, previous)


class BKTModel(KTModel):
    model_name = "BKT"

    def __init__(self, concept_count: int):
        super().__init__()
        self.concept_count = concept_count
        self.prior_logit = nn.Parameter(torch.full((concept_count,), -0.4))
        self.learn_logit = nn.Parameter(torch.full((concept_count,), -1.5))
        self.forget_logit = nn.Parameter(torch.full((concept_count,), -4.0))
        self.guess_logit = nn.Parameter(torch.full((concept_count,), -1.4))
        self.slip_logit = nn.Parameter(torch.full((concept_count,), -2.0))

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].float()
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        prior = torch.sigmoid(self.prior_logit)
        mastery = prior[None, :].expand(batch_size, -1).clone()
        learn = torch.sigmoid(self.learn_logit)
        forget = torch.sigmoid(self.forget_logit)
        guess = torch.sigmoid(self.guess_logit)
        slip = torch.sigmoid(self.slip_logit)
        logits: list[Tensor] = []
        for step in range(length):
            active = valid[:, step]
            concept = indices[:, step]
            current = mastery.gather(1, concept[:, None]).squeeze(1)
            g = guess[concept]
            s = slip[concept]
            probability = current * (1.0 - s) + (1.0 - current) * g
            probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
            logits.append(torch.logit(probability))
            response = labels[:, step]
            posterior_correct = current * (1.0 - s) / probability
            posterior_incorrect = current * s / (1.0 - probability)
            posterior = torch.where(response >= 0.5, posterior_correct, posterior_incorrect)
            transitioned = (posterior + (1.0 - posterior) * learn[concept]) * (
                1.0 - forget[concept]
            )
            replacement = mastery.scatter(1, concept[:, None], transitioned[:, None])
            mastery = _active_select(mastery, replacement, active)
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        return KTOutput(stacked, valid, zero, zero, {"final_mastery": mastery})


class RecurrentKT(KTModel):
    def __init__(
        self,
        model_name: str,
        concept_embeddings: np.ndarray | Tensor,
        hidden_size: int = 128,
        layers: int = 1,
        dropout: float = 0.2,
        semantic_input: bool = False,
        uncertainty: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        embeddings = torch.as_tensor(concept_embeddings, dtype=torch.float32)
        self.register_buffer("text_embeddings", embeddings)
        concept_count, text_size = embeddings.shape
        self.semantic_input = semantic_input
        self.uses_semantic_descriptors = semantic_input
        self.uncertainty = uncertainty
        if semantic_input:
            self.concept_projection = nn.Linear(text_size, hidden_size)
        else:
            self.concept_embedding = nn.Embedding(concept_count, hidden_size)
        self.response_embedding = nn.Embedding(2, hidden_size // 4)
        input_size = hidden_size + hidden_size // 4
        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        output_size = 2 if uncertainty else 1
        self.output = FeedForward(2 * hidden_size, output_size, [hidden_size], dropout)

    def concept_representation(self, indices: Tensor) -> Tensor:
        if self.semantic_input:
            return self.concept_projection(self.text_embeddings[indices])
        return self.concept_embedding(indices)

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].float()
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        hidden = [indices.new_zeros((batch_size, cell.hidden_size), dtype=torch.float32) for cell in self.cells]
        cell_state = [value.clone() for value in hidden]
        logits: list[Tensor] = []
        log_variances: list[Tensor] = []
        for step in range(length):
            active = valid[:, step]
            concept = self.concept_representation(indices[:, step])
            prediction = self.output(torch.cat([hidden[-1], concept], dim=-1))
            if self.uncertainty:
                log_variance = prediction[:, 1].clamp(-8.0, 5.0)
                adjusted = prediction[:, 0] / torch.sqrt(
                    1.0 + math.pi * torch.exp(log_variance) / 8.0
                )
                logits.append(adjusted)
                log_variances.append(log_variance)
            else:
                logits.append(prediction.squeeze(-1))
            response = self.response_embedding(labels[:, step].long().clamp(0, 1))
            layer_input = torch.cat([concept, response], dim=-1)
            for layer_index, recurrent in enumerate(self.cells):
                new_hidden, new_cell = recurrent(layer_input, (hidden[layer_index], cell_state[layer_index]))
                hidden[layer_index] = _active_select(hidden[layer_index], new_hidden, active)
                cell_state[layer_index] = _active_select(cell_state[layer_index], new_cell, active)
                layer_input = self.dropout(hidden[layer_index])
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        auxiliary: dict[str, Any] = {"final_hidden": hidden[-1]}
        if log_variances:
            auxiliary["log_variance"] = torch.stack(log_variances, dim=1)
        return KTOutput(stacked, valid, zero, zero, auxiliary)


class TransformerKT(KTModel):
    def __init__(
        self,
        model_name: str,
        concept_embeddings: np.ndarray | Tensor,
        hidden_size: int = 128,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.2,
        semantic_input: bool = False,
        distance_bias: str = "none",
        maximum_length: int = 50,
    ):
        super().__init__()
        self.model_name = model_name
        embeddings = torch.as_tensor(concept_embeddings, dtype=torch.float32)
        self.register_buffer("text_embeddings", embeddings)
        concept_count, text_size = embeddings.shape
        self.semantic_input = semantic_input
        self.uses_semantic_descriptors = semantic_input
        self.distance_bias = distance_bias
        self.maximum_length = maximum_length
        if semantic_input:
            self.concept_projection = nn.Linear(text_size, hidden_size)
        else:
            self.concept_embedding = nn.Embedding(concept_count, hidden_size)
        self.response_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Embedding(maximum_length + 1, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output = FeedForward(2 * hidden_size, 1, [hidden_size], dropout)

    def concept_representation(self, indices: Tensor) -> Tensor:
        if self.semantic_input:
            return self.concept_projection(self.text_embeddings[indices])
        return self.concept_embedding(indices)

    def attention_mask(self, length: int, device: torch.device) -> Tensor:
        if self.distance_bias == "none":
            return causal_mask(length, device)
        query = torch.arange(length, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        distance = query - key
        if self.distance_bias == "exponential":
            bias = -0.1 * distance.clamp_min(0).float()
        elif self.distance_bias == "cone_kernel":
            bias = -torch.sqrt(distance.clamp_min(0).float())
        else:
            raise ValueError(f"Unknown attention distance bias: {self.distance_bias}")
        return bias.masked_fill(distance < 0, float("-inf"))

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].long().clamp(0, 1)
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        concepts = self.concept_representation(indices)
        interactions = concepts + self.response_embedding(labels)
        shifted = torch.zeros_like(interactions)
        shifted[:, 1:] = interactions[:, :-1]
        positions = torch.arange(length, device=indices.device).clamp_max(self.maximum_length)
        shifted = shifted + self.position_embedding(positions)[None, :, :]
        shifted_valid = torch.ones_like(valid)
        shifted_valid[:, 1:] = valid[:, :-1]
        attention_mask = self.attention_mask(length, indices.device)
        padding_mask: Tensor = ~shifted_valid
        if attention_mask.dtype != torch.bool:
            padding_mask = padding_mask.float().masked_fill(padding_mask, float("-inf"))
        encoded = self.encoder(
            shifted,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )
        logits = self.output(torch.cat([encoded, concepts], dim=-1)).squeeze(-1)
        zero = zero_regularizer(logits)
        return KTOutput(logits, valid, zero, zero, {"encoded_history": encoded})


class DKVMNModel(KTModel):
    model_name = "DKVMN"

    def __init__(
        self,
        concept_embeddings: np.ndarray | Tensor,
        hidden_size: int = 128,
        memory_slots: int = 64,
        value_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        embeddings = torch.as_tensor(concept_embeddings, dtype=torch.float32)
        self.concept_embedding = nn.Embedding(embeddings.shape[0], hidden_size)
        self.keys = nn.Parameter(torch.randn(memory_slots, hidden_size) / math.sqrt(hidden_size))
        self.initial_values = nn.Parameter(torch.zeros(memory_slots, value_size))
        self.response_embedding = nn.Embedding(2, hidden_size // 4)
        self.erase = nn.Linear(hidden_size + hidden_size // 4, value_size)
        self.add = nn.Linear(hidden_size + hidden_size // 4, value_size)
        self.output = FeedForward(value_size + hidden_size, 1, [hidden_size], dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].long().clamp(0, 1)
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        values = self.initial_values[None, :, :].expand(batch_size, -1, -1).clone()
        logits: list[Tensor] = []
        for step in range(length):
            active = valid[:, step]
            concept = self.concept_embedding(indices[:, step])
            weights = torch.softmax(concept @ self.keys.T / math.sqrt(self.keys.shape[1]), dim=-1)
            read = torch.einsum("bm,bmd->bd", weights, values)
            logits.append(self.output(torch.cat([read, concept], dim=-1)).squeeze(-1))
            interaction = torch.cat([concept, self.response_embedding(labels[:, step])], dim=-1)
            erase = torch.sigmoid(self.erase(interaction))[:, None, :]
            add = torch.tanh(self.add(interaction))[:, None, :]
            write_weight = weights[:, :, None]
            updated = values * (1.0 - write_weight * erase) + write_weight * add
            values = _active_select(values, updated, active)
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        return KTOutput(stacked, valid, zero, zero, {"final_memory": values})


class DiscreteStateKT(KTModel):
    """Concept-indexed student state table used by the A4 field ablation."""

    model_name = "A4-DiscreteState"

    def __init__(
        self,
        concept_count: int,
        state_size: int = 256,
        capacity_size: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.concept_embedding = nn.Embedding(concept_count, state_size)
        self.initial_state = nn.Parameter(torch.zeros(concept_count, state_size))
        self.response_embedding = nn.Embedding(2, state_size // 4)
        update_input = 2 * state_size + state_size // 4
        self.update = FeedForward(
            update_input,
            state_size,
            [capacity_size, capacity_size, capacity_size],
            dropout,
        )
        self.output = FeedForward(
            2 * state_size, 1, [state_size], dropout
        )

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].long().clamp(0, 1)
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        state = self.initial_state[None, :, :].expand(batch_size, -1, -1).clone()
        logits: list[Tensor] = []
        rows = torch.arange(batch_size, device=indices.device)
        for step in range(length):
            active = valid[:, step]
            concept_index = indices[:, step]
            node = state[rows, concept_index]
            concept = self.concept_embedding(concept_index)
            logits.append(
                self.output(torch.cat([node, concept], dim=-1)).squeeze(-1)
            )
            response = self.response_embedding(labels[:, step])
            increment = torch.tanh(
                self.update(torch.cat([node, concept, response], dim=-1))
            )
            replacement = state.clone()
            replacement[rows, concept_index] = node + increment
            state = _active_select(state, replacement, active)
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        return KTOutput(stacked, valid, zero, zero, {"final_discrete_state": state})


class GraphKTModel(KTModel):
    model_name = "GKT"
    uses_semantic_descriptors = True

    def __init__(
        self,
        concept_embeddings: np.ndarray | Tensor,
        hidden_size: int = 128,
        graph_neighbors: int = 5,
        dropout: float = 0.2,
        capacity_size: int | None = None,
    ):
        super().__init__()
        embeddings = F.normalize(torch.as_tensor(concept_embeddings, dtype=torch.float32), dim=-1)
        self.register_buffer("text_embeddings", embeddings)
        count, text_size = embeddings.shape
        self.graph_neighbors = int(graph_neighbors)
        self._training_concepts: tuple[int, ...] = tuple(range(count))
        similarity = embeddings @ embeddings.T
        k = min(graph_neighbors + 1, count)
        nearest = torch.topk(similarity, k=k, dim=-1).indices
        adjacency = torch.zeros((count, count), dtype=torch.float32)
        for source in range(count):
            adjacency[source, nearest[source]] = F.relu(similarity[source, nearest[source]])
        adjacency = torch.maximum(adjacency, adjacency.T)
        adjacency.fill_diagonal_(1.0)
        adjacency /= adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        self.register_buffer("adjacency", adjacency)
        self.concept_projection = nn.Linear(text_size, hidden_size)
        self.initial_state = nn.Parameter(torch.zeros(count, hidden_size))
        self.response_embedding = nn.Embedding(2, hidden_size // 4)
        self.update = nn.GRUCell(hidden_size + hidden_size // 4, hidden_size)
        self.capacity_update = (
            FeedForward(
                2 * hidden_size + hidden_size // 4,
                hidden_size,
                [capacity_size, capacity_size, capacity_size],
                dropout,
            )
            if capacity_size is not None
            else None
        )
        self.message = nn.Linear(hidden_size, hidden_size)
        self.output = FeedForward(2 * hidden_size, 1, [hidden_size], dropout)

    @torch.no_grad()
    def _set_partition_adjacency(
        self, concept_indices: list[int], activate_unseen: bool
    ) -> None:
        count = int(self.text_embeddings.shape[0])
        seen = sorted(set(int(index) for index in concept_indices))
        if not seen:
            raise ValueError("GKT graph requires at least one training concept")
        unseen = sorted(set(range(count)) - set(seen))
        similarity = self.text_embeddings @ self.text_embeddings.T
        adjacency = torch.zeros_like(similarity)
        seen_tensor = torch.as_tensor(seen, dtype=torch.long, device=similarity.device)
        k_seen = min(self.graph_neighbors + 1, len(seen))
        for source in seen:
            scores = similarity[source, seen_tensor]
            nearest_local = torch.topk(scores, k=k_seen).indices
            targets = seen_tensor[nearest_local]
            adjacency[source, targets] = F.relu(similarity[source, targets])
        if activate_unseen:
            k_insert = min(self.graph_neighbors, len(seen))
            for source in unseen:
                nearest_local = torch.topk(
                    similarity[source, seen_tensor], k=k_insert
                ).indices
                targets = seen_tensor[nearest_local]
                weights = F.relu(similarity[source, targets])
                adjacency[source, targets] = weights
                adjacency[targets, source] = weights
                self.initial_state[source].copy_(self.initial_state[targets].mean(dim=0))
        adjacency = torch.maximum(adjacency, adjacency.T)
        adjacency.fill_diagonal_(1.0)
        adjacency /= adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        self.adjacency.copy_(adjacency)

    def configure_training_concepts(self, concept_indices: list[int]) -> None:
        self._training_concepts = tuple(sorted(set(int(index) for index in concept_indices)))
        self._set_partition_adjacency(list(self._training_concepts), activate_unseen=False)

    def activate_unseen_concepts(self) -> None:
        self._set_partition_adjacency(list(self._training_concepts), activate_unseen=True)

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].long().clamp(0, 1)
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        state = self.initial_state[None, :, :].expand(batch_size, -1, -1).clone()
        logits: list[Tensor] = []
        for step in range(length):
            active = valid[:, step]
            concept_index = indices[:, step]
            row = torch.arange(batch_size, device=indices.device)
            node = state[row, concept_index]
            semantic = self.concept_projection(self.text_embeddings[concept_index])
            logits.append(self.output(torch.cat([node, semantic], dim=-1)).squeeze(-1))
            interaction = torch.cat([semantic, self.response_embedding(labels[:, step])], dim=-1)
            new_node = (
                node
                + torch.tanh(
                    self.capacity_update(
                        torch.cat([node, interaction], dim=-1)
                    )
                )
                if self.capacity_update is not None
                else self.update(interaction, node)
            )
            local = state.clone()
            local[row, concept_index] = new_node
            propagated = torch.einsum("mn,bnd->bmd", self.adjacency, self.message(local))
            updated = 0.5 * local + 0.5 * torch.tanh(propagated)
            state = _active_select(state, updated, active)
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        return KTOutput(stacked, valid, zero, zero, {"final_node_state": state})


class SemanticAttributeKT(RecurrentKT):
    def __init__(
        self,
        model_name: str,
        concept_embeddings: np.ndarray | Tensor,
        hidden_size: int,
        dropout: float,
        relation_heads: int = 1,
    ):
        super().__init__(
            model_name,
            concept_embeddings,
            hidden_size=hidden_size,
            layers=1,
            dropout=dropout,
            semantic_input=True,
        )
        text_size = int(self.text_embeddings.shape[1])
        self.attribute_gate = nn.Sequential(
            nn.Linear(text_size, hidden_size), nn.Sigmoid()
        )
        self.relation_heads = max(1, relation_heads)
        self.relation_projection = nn.Linear(text_size, hidden_size * self.relation_heads)

    def concept_representation(self, indices: Tensor) -> Tensor:
        text = self.text_embeddings[indices]
        base = self.concept_projection(text)
        gate = self.attribute_gate(text)
        relations = self.relation_projection(text).reshape(
            *text.shape[:-1], self.relation_heads, base.shape[-1]
        )
        relation = relations.mean(dim=-2)
        return gate * base + (1.0 - gate) * torch.tanh(relation)


class MetaRecurrentKT(RecurrentKT):
    model_name = "MAML-KT"

    def __init__(
        self,
        concept_embeddings: np.ndarray | Tensor,
        hidden_size: int = 128,
        layers: int = 1,
        dropout: float = 0.2,
        inner_steps: int = 3,
        inner_learning_rate: float = 1.0e-2,
    ):
        super().__init__(
            "MAML-KT",
            concept_embeddings,
            hidden_size=hidden_size,
            layers=layers,
            dropout=dropout,
            semantic_input=True,
        )
        self.inner_steps = int(inner_steps)
        self.inner_learning_rate = float(inner_learning_rate)

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].float()
        valid = batch["valid_mask"].bool()
        batch_size, length = indices.shape
        hidden = [
            indices.new_zeros((batch_size, cell.hidden_size), dtype=torch.float32)
            for cell in self.cells
        ]
        cell_state = [value.clone() for value in hidden]
        fast_bias = indices.new_zeros((batch_size,), dtype=torch.float32)
        logits: list[Tensor] = []
        for step in range(length):
            active = valid[:, step]
            concept = self.concept_representation(indices[:, step])
            base_logit = self.output(torch.cat([hidden[-1], concept], dim=-1)).squeeze(-1)
            logit = base_logit + fast_bias
            logits.append(logit)
            adapted = fast_bias
            for _ in range(self.inner_steps):
                adapted = adapted + self.inner_learning_rate * (
                    labels[:, step] - torch.sigmoid(base_logit + adapted)
                )
            fast_bias = torch.where(active, adapted, fast_bias)
            response = self.response_embedding(labels[:, step].long().clamp(0, 1))
            layer_input = torch.cat([concept, response], dim=-1)
            for layer_index, recurrent in enumerate(self.cells):
                new_hidden, new_cell = recurrent(
                    layer_input, (hidden[layer_index], cell_state[layer_index])
                )
                hidden[layer_index] = _active_select(
                    hidden[layer_index], new_hidden, active
                )
                cell_state[layer_index] = _active_select(
                    cell_state[layer_index], new_cell, active
                )
                layer_input = self.dropout(hidden[layer_index])
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        return KTOutput(
            stacked,
            valid,
            zero,
            zero,
            {"final_hidden": hidden[-1], "fast_bias": fast_bias},
        )


class DisentangledKT(RecurrentKT):
    model_name = "DisenKT"

    def __init__(self, concept_embeddings: np.ndarray | Tensor, hidden_size: int, dropout: float):
        super().__init__(
            "DisenKT",
            concept_embeddings,
            hidden_size=hidden_size,
            layers=1,
            dropout=dropout,
            semantic_input=True,
        )
        self.shared_projection = nn.Linear(hidden_size, hidden_size)
        self.private_projection = nn.Linear(hidden_size, hidden_size)
        self.domain_gate = nn.Parameter(torch.zeros(16, hidden_size))

    def concept_representation(self, indices: Tensor) -> Tensor:
        base = super().concept_representation(indices)
        return torch.tanh(self.shared_projection(base)) + torch.tanh(self.private_projection(base))

    def _domain_indices(self, datasets: list[str], device: torch.device) -> Tensor:
        values = [
            int.from_bytes(
                hashlib.blake2b(str(dataset).encode("utf-8"), digest_size=4).digest(),
                "little",
            )
            % self.domain_gate.shape[0]
            for dataset in datasets
        ]
        return torch.as_tensor(values, dtype=torch.long, device=device)

    def forward(self, batch: Mapping[str, Tensor]) -> KTOutput:
        indices = batch["concept_indices"].long()
        labels = batch["labels"].float()
        valid = batch["valid_mask"].bool()
        datasets = [str(value) for value in batch["dataset"]]  # type: ignore[index]
        domain = self.domain_gate[self._domain_indices(datasets, indices.device)]
        batch_size, length = indices.shape
        hidden = [
            indices.new_zeros((batch_size, cell.hidden_size), dtype=torch.float32)
            for cell in self.cells
        ]
        cell_state = [value.clone() for value in hidden]
        logits: list[Tensor] = []
        for step in range(length):
            active = valid[:, step]
            base = RecurrentKT.concept_representation(self, indices[:, step])
            shared = torch.tanh(self.shared_projection(base))
            private = torch.tanh(self.private_projection(base + domain))
            concept = shared + private
            logits.append(
                self.output(torch.cat([hidden[-1], concept], dim=-1)).squeeze(-1)
            )
            response = self.response_embedding(labels[:, step].long().clamp(0, 1))
            layer_input = torch.cat([concept, response], dim=-1)
            for layer_index, recurrent in enumerate(self.cells):
                new_hidden, new_cell = recurrent(
                    layer_input, (hidden[layer_index], cell_state[layer_index])
                )
                hidden[layer_index] = _active_select(
                    hidden[layer_index], new_hidden, active
                )
                cell_state[layer_index] = _active_select(
                    cell_state[layer_index], new_cell, active
                )
                layer_input = self.dropout(hidden[layer_index])
        stacked = torch.stack(logits, dim=1)
        zero = zero_regularizer(stacked)
        return KTOutput(stacked, valid, zero, zero, {"final_hidden": hidden[-1]})
