"""Texto → traços de tinta (fonte vetorial Hershey), para desenhar resultados na página.

Usado por POST /page/process: o resultado da conta é inserido no Xournal++ como *traços*
(app.addStrokes), estilo iPad Math Notes — não como caixa de texto. Glifos em _hershey.py
(gerado por scripts/gen_inkfont_glyphs.py; domínio público).
"""

from __future__ import annotations

import unicodedata

from ._hershey import GLYPHS


def strip_accents(text: str) -> str:
    """A fonte Hershey só tem ASCII: 'círculo' → 'circulo', '·' → '-'."""
    text = text.replace("·", "-")
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

# Extensão vertical dos dígitos na fonte (y de -12 a 9, y cresce para baixo).
_CAP_TOP = -12.0
_CAP_BOTTOM = 9.0
_CAP_HEIGHT = _CAP_BOTTOM - _CAP_TOP
_CAP_MID = (_CAP_TOP + _CAP_BOTTOM) / 2.0

Polyline = list[tuple[float, float]]


def text_to_strokes(text: str, x: float, y_mid: float, height: float) -> list[Polyline]:
    """Desenha `text` começando em `x`, centrado verticalmente em `y_mid`.

    `height` é a altura de maiúsculas/dígitos desejada (na prática: a altura da escrita da
    conta). Caracteres sem glifo viram '?'. Retorna polilinhas em coordenadas da página.
    """
    if height <= 0:
        raise ValueError("height deve ser > 0")
    scale = height / _CAP_HEIGHT
    out: list[Polyline] = []
    cursor = x
    for ch in text:
        left, right, strokes = GLYPHS.get(ch) or GLYPHS["?"]
        for stroke in strokes:
            out.append(
                [
                    (cursor + (gx - left) * scale, y_mid + (gy - _CAP_MID) * scale)
                    for gx, gy in stroke
                ]
            )
        cursor += (right - left) * scale
    return out


def text_width(text: str, height: float) -> float:
    """Largura que `text` ocupará com a mesma métrica de text_to_strokes."""
    scale = height / _CAP_HEIGHT
    return sum(
        ((g := GLYPHS.get(ch) or GLYPHS["?"])[1] - g[0]) * scale for ch in text
    )
