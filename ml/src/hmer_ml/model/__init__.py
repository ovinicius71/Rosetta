"""Modelo: encoder de tinta compartilhado + cabeças plugáveis. Ver ADR 0006."""

from .encoder import build_encoder
from .heads import build_head
from .seq2seq import InkModel, build_model

__all__ = ["InkModel", "build_model", "build_encoder", "build_head"]
