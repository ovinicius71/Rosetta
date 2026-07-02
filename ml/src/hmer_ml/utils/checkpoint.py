"""Checkpoint com resume — treinos são fatiados em várias sessões (restrição de hardware).

Salva modelo + otimizador + scheduler + scaler(AMP) + passo/época, para retomar bit-a-bit.
Ver docs/roadmap.md (Fase 1).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, *, model, optimizer, scheduler, scaler, step: int,
                    epoch: int, cfg=None, extra: dict | None = None) -> None:
    """Salva estado completo p/ resume. Escrita atômica (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None,
                    scaler=None, map_location="cpu") -> tuple[int, int]:
    """Carrega estado; retorna (step, epoch). optimizer/scheduler opcionais (inferência)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0)


def latest_checkpoint(dirpath: str | Path) -> Path | None:
    """Retorna o checkpoint mais recente do diretório, ou None."""
    d = Path(dirpath)
    if not d.exists():
        return None
    ckpts = sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None
