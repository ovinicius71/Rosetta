"""Carregamento preguiçoso do Recognizer (modelo de ML) e ponte com a rota /recognize.

Mantém uma instância única (o modelo carrega uma vez, no primeiro request). Configurável
por variáveis de ambiente:

  HMER_CKPT    caminho do checkpoint (.ckpt). Sem ele → 501 (modo stub p/ dev do front).
  HMER_CONFIG  config do TREINO correspondente ao checkpoint (arquitetura + vocab_path).
               Default: ml/configs/crohme.yaml
  HMER_DEVICE  cpu | cuda. Default: cpu — inferência do modelo compacto é rápida em CPU
               e não disputa a GPU com um treino em andamento.

Rode o uvicorn a partir da RAIZ do repo (C:\\HMER): os caminhos do config (vocab_path)
são relativos a ela.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import HTTPException


@lru_cache(maxsize=1)
def get_recognizer():
    """Instancia hmer_ml.infer.Recognizer uma vez (ou o stub, sem checkpoint)."""
    ckpt = os.getenv("HMER_CKPT")
    config = os.getenv("HMER_CONFIG", "ml/configs/crohme.yaml")
    device = os.getenv("HMER_DEVICE", "cpu")

    if not ckpt:
        return _StubRecognizer()
    if not os.path.exists(ckpt):
        raise HTTPException(status_code=500, detail=f"HMER_CKPT não encontrado: {ckpt}")

    from hmer_ml.infer import Recognizer

    print(f"[api] carregando modelo: ckpt={ckpt} config={config} device={device}")
    return _ModelRecognizer(Recognizer(ckpt, config_path=config, device=device))


class _ModelRecognizer:
    """Adaptador fino: aplica defaults de inferência e traduz erros p/ HTTP."""

    def __init__(self, recognizer):
        self._rec = recognizer
        infer_cfg = recognizer.cfg.get("infer", {})
        self.beam_size = infer_cfg.get("beam_size", 4)
        self.max_len = infer_cfg.get("max_len", 256)

    def recognize(self, ink: dict) -> str:
        strokes = ink.get("strokes") or []
        if not any(s.get("points") for s in strokes):
            raise HTTPException(status_code=422, detail="tinta vazia (nenhum ponto)")
        try:
            return self._rec.recognize(ink, beam_size=self.beam_size, max_len=self.max_len)
        except Exception as e:  # noqa: BLE001 - erro de inferência não deve derrubar a API
            raise HTTPException(status_code=500, detail=f"falha na inferência: {e}") from e


class _StubRecognizer:
    def recognize(self, ink: dict, **kw) -> str:
        raise HTTPException(
            status_code=501,
            detail="Modelo não carregado. Defina HMER_CKPT (ex.: checkpoints/crohme/last.ckpt).",
        )
