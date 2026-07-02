"""Cabeças de saída plugáveis sobre o encoder de tinta compartilhado. Ver ADR 0006.

- LatexHead     -> decoder Transformer autorregressivo (matemática) — AGORA.
- SketchClsHead -> classificador de categoria (QuickDraw) — FUTURO (Fase 4).

Ambas consomem a `memory` produzida pelo InkEncoder e NADA sabem sobre como a tinta foi
codificada. Trocar de tarefa = trocar de cabeça, não de pipeline de entrada.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import PositionalEncoding, causal_mask


class LatexHead(nn.Module):
    """Transformer decoder autorregressivo → tokens de LaTeX.

    Treino: teacher forcing (recebe o alvo deslocado). Inferência: greedy/beam (infer.py).
    """

    def __init__(self, d_model: int, vocab_size: int, layers: int, heads: int,
                 ff: int, dropout: float, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, memory, memory_key_padding_mask, tgt_in):
        """memory [B,T,D], mask [B,T] (True=pad), tgt_in [B,L] → logits [B,L,vocab]."""
        L = tgt_in.size(1)
        tgt_emb = self.pos(self.embed(tgt_in))
        tgt_mask = causal_mask(L, device=tgt_in.device)
        tgt_pad = tgt_in == self.pad_id
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.proj(out)


class SketchClsHead(nn.Module):
    """FUTURO (Fase 4): pooling da memory + MLP → logits de categoria de desenho.

    Deixado explícito para provar que o encoder é reutilizável sem alterações.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, memory, memory_key_padding_mask):
        # masked mean pooling sobre os passos válidos
        valid = (~memory_key_padding_mask).unsqueeze(-1).float()
        pooled = (memory * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return self.proj(pooled)


def build_head(cfg, vocab_size: int | None = None, pad_id: int | None = None):
    """Fábrica de cabeça pela config (head: latex | sketch_cls)."""
    head = cfg.model.head
    if head == "latex":
        if vocab_size is None or pad_id is None:
            raise ValueError("LatexHead requer vocab_size e pad_id")
        d = cfg.model.decoder
        return LatexHead(
            d_model=cfg.model.d_model,
            vocab_size=vocab_size,
            layers=d.layers,
            heads=d.heads,
            ff=d.ff,
            dropout=d.get("dropout", 0.1),
            pad_id=pad_id,
        )
    if head == "sketch_cls":  # Fase 4
        return SketchClsHead(cfg.model.d_model, cfg.model.get("num_classes", 345))
    raise ValueError(f"model.head desconhecido: {head}")
