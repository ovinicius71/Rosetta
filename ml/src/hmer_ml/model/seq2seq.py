"""InkModel = encoder de tinta (compartilhado) + cabeça (plugável). Ver ADR 0006.

Único lugar que junta encoder + head. `build_model(cfg)` lê a config e monta a combinação
certa (head: latex agora; sketch_cls na Fase 4) — sem que o encoder saiba da tarefa.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import build_encoder
from .heads import build_head


class InkModel(nn.Module):
    def __init__(self, encoder, head, *, bos_id: int | None = None, eos_id: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.bos_id = bos_id
        self.eos_id = eos_id

    def encode(self, batch):
        return self.encoder(batch["src"], batch["src_lengths"])

    def forward(self, batch):
        """Teacher forcing (head latex): consome tgt[:, :-1], prevê tgt[:, 1:].

        Retorna logits [B, L-1, vocab]. A perda é calculada no train.py.
        """
        memory, mmask = self.encode(batch)
        if isinstance(self.head, _LATEX_HEAD_TYPES):
            tgt_in = batch["tgt"][:, :-1]
            return self.head(memory, mmask, tgt_in)
        # sketch_cls e outras cabeças não-autorregressivas:
        return self.head(memory, mmask)

    @torch.no_grad()
    def greedy_decode(self, batch, max_len: int = 256) -> list[list[int]]:
        """Decodifica greedy a partir da tinta. Retorna ids (sem <bos>, corta em <eos>)."""
        assert self.bos_id is not None and self.eos_id is not None
        self.eval()
        memory, mmask = self.encode(batch)
        b = memory.size(0)
        device = memory.device
        ys = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = self.head(memory, mmask, ys)  # [B, cur, vocab]
            nxt = logits[:, -1].argmax(-1)  # [B]
            nxt = nxt.masked_fill(done, self.eos_id)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == self.eos_id)
            if bool(done.all()):
                break
        # remove <bos> e corta em <eos>
        out = []
        for row in ys[:, 1:].tolist():
            seq = []
            for t in row:
                if t == self.eos_id:
                    break
                seq.append(t)
            out.append(seq)
        return out


# Cabeças autorregressivas (recebem tgt_in). Import tardio para evitar ciclo.
from .heads import LatexHead  # noqa: E402

_LATEX_HEAD_TYPES = (LatexHead,)


def build_model(cfg, vocab_size: int | None = None, pad_id: int | None = None,
                bos_id: int | None = None, eos_id: int | None = None) -> InkModel:
    """Monta o modelo a partir da config. Ponto único de composição encoder+head."""
    encoder = build_encoder(cfg)
    head = build_head(cfg, vocab_size=vocab_size, pad_id=pad_id)
    return InkModel(encoder, head, bos_id=bos_id, eos_id=eos_id)
