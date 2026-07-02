"""Encoder de tinta — AGNÓSTICO À TAREFA. Único ponto de reuso entre matemática e desenhos.

REGRA (ADR 0006): nenhum símbolo de LaTeX / conceito de tarefa pode entrar aqui. A
interface é `forward(features, lengths) -> (memory, memory_key_padding_mask)`. Serve tanto
ao decoder LaTeX quanto, no futuro, ao classificador de desenhos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """[B, max_len] bool: True nas posições de padding (>= comprimento real)."""
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ar >= lengths.unsqueeze(1)


class InkEncoder(nn.Module):
    """Base comum. Implementações concretas: BiGRUEncoder, TransformerInkEncoder."""

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        raise NotImplementedError


class BiGRUEncoder(InkEncoder):
    """Encoder recorrente (default). Cabe folgado em VRAM ~6-8 GB. Ver ADR 0002."""

    def __init__(self, in_features: int, d_model: int, hidden: int, layers: int,
                 dropout: float, bidirectional: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(out_dim, d_model)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        # features: [B, T, F]; lengths: [B]
        x = self.input_proj(features)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        memory = self.output_proj(out)  # [B, T, d_model]
        mask = _padding_mask(lengths, memory.size(1))
        return memory, mask


class TransformerInkEncoder(InkEncoder):
    """Alternativa escalável (mais VRAM). Selecionada por config encoder.type=transformer."""

    def __init__(self, in_features: int, d_model: int, layers: int, heads: int,
                 ff: int, dropout: float):
        super().__init__()
        from .decoder import PositionalEncoding

        self.input_proj = nn.Linear(in_features, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        x = self.pos(self.input_proj(features))
        mask = _padding_mask(lengths, x.size(1))
        memory = self.encoder(x, src_key_padding_mask=mask)
        return memory, mask


def build_encoder(cfg) -> InkEncoder:
    """Fábrica: escolhe a implementação pela config (bigru | transformer)."""
    m = cfg.model
    etype = m.encoder.type
    if etype == "bigru":
        return BiGRUEncoder(
            in_features=m.in_features,
            d_model=m.d_model,
            hidden=m.encoder.hidden,
            layers=m.encoder.layers,
            dropout=m.encoder.get("dropout", 0.1),
            bidirectional=m.encoder.get("bidirectional", True),
        )
    if etype == "transformer":
        return TransformerInkEncoder(
            in_features=m.in_features,
            d_model=m.d_model,
            layers=m.encoder.layers,
            heads=m.encoder.get("heads", 4),
            ff=m.encoder.get("ff", 4 * m.d_model),
            dropout=m.encoder.get("dropout", 0.1),
        )
    raise ValueError(f"encoder.type desconhecido: {etype}")
