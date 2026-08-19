"""Text encoders, content-addressed caches, and controlled fine-tuning."""

from .encoder import TextEncoder, build_text_encoder, encode_with_cache
from .trainable import TrainableDescriptorEncoder

__all__ = [
    "TextEncoder",
    "TrainableDescriptorEncoder",
    "build_text_encoder",
    "encode_with_cache",
]
