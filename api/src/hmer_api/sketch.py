"""POST /sketch/recognize: desenho do caderno → categoria + rótulo desenhado em tinta.

Espelha o padrão de recognize.py (instância única via lru_cache, envs, 501 sem
checkpoint). O rótulo em pt-BR volta também como traços Hershey (inkfont) posicionados
logo abaixo do desenho, na cor de resultado — mesma linguagem visual das contas.
"""

from __future__ import annotations

import os
import unicodedata
from functools import lru_cache

from fastapi import HTTPException

from hmer_ml.data.ink import Ink, Point, Stroke
from hmer_ml.segment import RESULT_COLOR

from .inkfont import text_to_strokes, text_width
from .schemas import PageInk, PageStroke, SketchGuess, SketchRecognizeResponse

DEFAULT_CKPT = "checkpoints/quickdraw/best.ckpt"

# categoria QuickDraw → rótulo exibido (pt-BR). Acentos são removidos só na tinta.
LABELS_PT = {
    "cat": "gato", "dog": "cachorro", "house": "casa", "tree": "árvore", "sun": "sol",
    "car": "carro", "flower": "flor", "fish": "peixe", "bird": "pássaro",
    "star": "estrela", "clock": "relógio", "bicycle": "bicicleta", "airplane": "avião",
    "sailboat": "barco", "cup": "xícara", "chair": "cadeira", "apple": "maçã",
    "moon": "lua", "umbrella": "guarda-chuva", "butterfly": "borboleta",
    "circle": "círculo",
}

# Posição/tamanho do rótulo relativo ao desenho
LABEL_HEIGHT_FRAC = 0.18  # altura do texto vs altura do desenho
LABEL_MIN_H = 13.0
LABEL_MAX_H = 26.0
LABEL_GAP_FRAC = 0.9  # distância do rótulo abaixo do desenho, em alturas de texto


@lru_cache(maxsize=1)
def get_sketch_recognizer():
    # SKETCH_CKPT: caminho explícito; "" desliga (testes); ausente → default se existir
    env = os.getenv("SKETCH_CKPT")
    if env is not None:
        ckpt = env or None
    else:
        ckpt = DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else None
    config = os.getenv("SKETCH_CONFIG", "ml/configs/quickdraw.yaml")
    device = os.getenv("HMER_DEVICE", "cpu")
    if not ckpt:
        return None
    if not os.path.exists(ckpt):
        raise HTTPException(status_code=500, detail=f"SKETCH_CKPT não encontrado: {ckpt}")

    from hmer_ml.infer_sketch import SketchRecognizer

    print(f"[api] carregando classificador de desenhos: ckpt={ckpt} device={device}")
    return SketchRecognizer(ckpt, config_path=config, device=device)


def _strip_accents(text: str) -> str:
    """A fonte Hershey só tem ASCII: 'círculo' → 'circulo'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def recognize_sketch(page: PageInk) -> SketchRecognizeResponse:
    rec = get_sketch_recognizer()
    if rec is None:
        raise HTTPException(
            status_code=501,
            detail="Classificador de desenhos não carregado. Treine (hmer_ml.train_sketch) "
            f"ou defina SKETCH_CKPT (default: {DEFAULT_CKPT}).",
        )
    if not page.strokes:
        raise HTTPException(status_code=422, detail="desenho vazio (nenhum traço)")

    ink = Ink(
        strokes=[Stroke(points=[Point(x, y) for x, y in zip(s.x, s.y)]) for s in page.strokes]
    )
    try:
        top = rec.recognize(ink, topk=3)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - erro de inferência não derruba a API
        raise HTTPException(status_code=500, detail=f"falha na classificação: {e}") from e

    guesses = [
        SketchGuess(label=lab, label_pt=LABELS_PT.get(lab, lab), confidence=round(p, 4))
        for lab, p in top
    ]
    best = guesses[0]
    return SketchRecognizeResponse(
        label=best.label,
        label_pt=best.label_pt,
        confidence=best.confidence,
        topk=guesses,
        strokes=_label_strokes(best.label_pt, page),
    )


def _label_strokes(label_pt: str, page: PageInk) -> list[PageStroke]:
    """Escreve o rótulo (sem acentos) centrado logo abaixo do desenho."""
    xs = [x for s in page.strokes for x in s.x]
    ys = [y for s in page.strokes for y in s.y]
    x0, x1, y1 = min(xs), max(xs), max(ys)
    h = max(LABEL_MIN_H, min((y1 - min(ys)) * LABEL_HEIGHT_FRAC, LABEL_MAX_H))
    text = _strip_accents(label_pt)
    x = (x0 + x1) / 2.0 - text_width(text, h) / 2.0
    y_mid = y1 + LABEL_GAP_FRAC * h + h / 2.0

    widths = sorted(s.width for s in page.strokes if s.width is not None)
    pen = widths[len(widths) // 2] if widths else 1.41

    return [
        PageStroke(x=[p[0] for p in poly], y=[p[1] for p in poly],
                   color=RESULT_COLOR, width=pen)
        for poly in text_to_strokes(text, x=x, y_mid=y_mid, height=h)
    ]
