"""Pinned sentence encoders plus a deterministic offline test encoder."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from semopkt.data.schema import normalize_text
from semopkt.utils.hashing import hash_file, hash_json
from semopkt.utils.io import ensure_parent, read_json, write_json


class TextEncoder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HashTextEncoder(TextEncoder):
    output_dimension: int = 64
    salt: str = "semopkt-hash-v1"

    @property
    def dimension(self) -> int:
        return self.output_dimension

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "hash",
            "model_id": "deterministic-token-hash-v1",
            "revision": self.salt,
            "dimension": self.dimension,
            "pooling": "signed-token-sum",
            "normalize": True,
            "total_parameters": 0,
            "trainable_parameters": 0,
        }

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        del batch_size
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = normalize_text(text)
            tokens = normalized.split() or [normalized]
            for token in tokens:
                digest = hashlib.blake2b(
                    f"{self.salt}:{token}".encode("utf-8"), digest_size=32
                ).digest()
                for offset in range(0, len(digest), 2):
                    index = int.from_bytes(digest[offset : offset + 2], "little") % self.dimension
                    matrix[row, index] += 1.0 if digest[offset] % 2 == 0 else -1.0
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerTextEncoder(TextEncoder):
    def __init__(self, config: Mapping[str, Any], device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for the configured text encoder"
            ) from error
        self._config = dict(config)
        self._model = SentenceTransformer(
            str(config["model_id"]),
            revision=str(config["revision"]),
            device=device,
            trust_remote_code=False,
        )
        self._model.max_seq_length = int(config.get("max_length", 128))
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._model.eval()
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        self._total_parameters = sum(parameter.numel() for parameter in self._model.parameters())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "sentence_transformers",
            "model_id": self._config["model_id"],
            "revision": self._config["revision"],
            "license": self._config.get("license"),
            "dimension": self.dimension,
            "max_length": int(self._config.get("max_length", 128)),
            "pooling": self._config.get("pooling", "mean"),
            "normalize": bool(self._config.get("normalize", True)),
            "instruction_prefix": self._config.get("instruction_prefix", ""),
            "cache_normalization": self._config.get(
                "cache_normalization", "unicode_nfkc_casefold_whitespace"
            ),
            "total_parameters": self._total_parameters,
            "trainable_parameters": 0,
        }

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        prefix = str(self._config.get("instruction_prefix", ""))
        normalized = [prefix + normalize_text(text) for text in texts]
        matrix = self._model.encode(
            normalized,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=bool(self._config.get("normalize", True)),
        )
        return np.asarray(matrix, dtype=np.float32)


def build_text_encoder(config: Mapping[str, Any], device: str | None = None) -> TextEncoder:
    backend = str(config.get("backend", "sentence_transformers"))
    if backend == "hash":
        dimension = int(config.get("dimension", config.get("output_dimension", 64)))
        revision = str(config.get("revision", "semopkt-hash-v1"))
        return HashTextEncoder(dimension, revision)
    if backend == "sentence_transformers":
        return SentenceTransformerTextEncoder(config, device=device)
    raise ValueError(f"Unsupported text encoder backend: {backend}")


def _cache_paths(cache_root: str | Path, key: str) -> tuple[Path, Path]:
    root = Path(cache_root)
    return root / f"{key}.npz", root / f"{key}.json"


def encode_with_cache(
    texts: Sequence[str],
    encoder: TextEncoder,
    cache_root: str | Path,
    namespace: str,
    batch_size: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    normalized = sorted(set(normalize_text(text) for text in texts))
    key_payload = {
        "namespace": namespace,
        "encoder": dict(encoder.metadata),
        "texts": normalized,
    }
    key = hash_json(key_payload)
    matrix_path, metadata_path = _cache_paths(cache_root, key)
    if matrix_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        with np.load(matrix_path, allow_pickle=False) as archive:
            matrix = archive["embeddings"]
        if (
            metadata.get("cache_key") != key
            or matrix.shape[0] != len(normalized)
            or metadata.get("matrix_sha256") != hash_file(matrix_path)
        ):
            raise ValueError(f"Embedding cache integrity failure: {matrix_path}")
    else:
        matrix = encoder.encode(normalized, batch_size=batch_size)
        if matrix.shape != (len(normalized), encoder.dimension):
            raise ValueError(
                f"Encoder returned shape {matrix.shape}; expected {(len(normalized), encoder.dimension)}"
            )
        ensure_parent(matrix_path)
        np.savez_compressed(matrix_path, embeddings=matrix.astype(np.float32))
        metadata = {
            "cache_key": key,
            "namespace": namespace,
            "encoder": dict(encoder.metadata),
            "text_count": len(normalized),
            "embedding_shape": list(matrix.shape),
            "text_sha256": hash_json(normalized),
            "matrix_sha256": hash_file(matrix_path),
        }
        write_json(metadata_path, metadata)
    lookup = {text: matrix[index].copy() for index, text in enumerate(normalized)}
    return lookup, metadata
