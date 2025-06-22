"""
标题生成器模块
"""

from .transformer_encoder import TransformerEncoder
from .pointer_decoder import PointerDecoder

__all__ = [
    "TransformerEncoder",
    "PointerDecoder"
]