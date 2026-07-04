"""Inferência do classificador de desenhos (Fase 4): tinta → categoria + confiança.

Espelha infer.Recognizer: carrega config + checkpoint uma vez e expõe `recognize`.
O pré-processamento é idêntico ao do treino (prepare_sketch: normalize → resample fixo),
então a densidade da tinta de entrada (caneta densa do caderno vs RDP esparso do
QuickDraw) não importa — por construção.
"""

from __future__ import annotations

import torch

from .data.ink import Ink
from .data.quickdraw import prepare_sketch
from .model import build_model
from .utils.checkpoint import load_checkpoint
from .utils.config import load_config


class SketchRecognizer:
    def __init__(self, ckpt_path: str, config_path: str, device: str = "cpu"):
        self.cfg = load_config(config_path)
        self.categories: list[str] = list(self.cfg.data.categories)
        self.device = device
        self.model = build_model(self.cfg).to(device)
        load_checkpoint(ckpt_path, model=self.model, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def recognize(self, ink: Ink | dict, topk: int = 3) -> list[tuple[str, float]]:
        """Tinta → [(categoria, probabilidade)] em ordem decrescente (topk)."""
        if isinstance(ink, dict):
            ink = Ink.from_dict(ink)
        feats = prepare_sketch(
            ink,
            max_points=self.cfg.data.get("max_points", 512),
            resample_step=self.cfg.data.get("resample_step"),
        )
        if not feats:
            raise ValueError("tinta vazia")
        src = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        lengths = torch.tensor([src.size(1)], dtype=torch.long, device=self.device)
        logits = self.model({"src": src, "src_lengths": lengths})[0]  # [C]
        probs = torch.softmax(logits, dim=-1)
        k = min(topk, probs.numel())
        top = torch.topk(probs, k)
        return [
            (self.categories[i], float(p))
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
