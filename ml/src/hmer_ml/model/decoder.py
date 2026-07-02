"""Blocos de decodificação reutilizados por LatexHead.

Separado de heads.py para manter a cabeça enxuta e permitir reuso (ex.: futuras cabeças
seq2seq para outras notações). Ver ADR 0006.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding senoidal (Vaswani et al.), somado aos embeddings de token."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        return self.dropout(x + self.pe[:, : x.size(1)])


def causal_mask(size: int, device=None) -> torch.Tensor:
    """Máscara causal booleana [L, L]: True nas posições futuras (proibidas de atender).

    Bool (não aditiva float) para casar o tipo com `tgt_key_padding_mask` (também bool) e
    evitar o aviso de máscaras mistas do nn.TransformerDecoder.
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
