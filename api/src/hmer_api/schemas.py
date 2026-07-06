"""Modelos Pydantic — espelho de schemas/ink.schema.json (ADR 0004).

Mantenha em sincronia com o JSON Schema e com web/lib/ink.ts e ml/data/ink.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Point(BaseModel):
    x: float
    y: float
    t: float | None = Field(default=None, description="ms desde o 1º ponto da tinta")


class Stroke(BaseModel):
    points: list[Point] = Field(min_length=1)


class Ink(BaseModel):
    """Corpo de POST /recognize. Igual ao contrato compartilhado."""

    schema_version: str = "1.0"
    width: float | None = None
    height: float | None = None
    strokes: list[Stroke]
    label: str | None = None  # ignorado na inferência


class RecognizeResponse(BaseModel):
    latex: str


class EvaluateRequest(BaseModel):
    latex: str


class EvaluateResponse(BaseModel):
    result: str | None = None
    error: str | None = None


# ---- integração Xournal++ (POST /page/process) -------------------------------------------
# Formato próprio (não é o contrato ink.schema.json): espelha o retorno de
# app.getStrokes("layer") do plugin Lua — arrays paralelos x[]/y[] + estilo por traço.


class PageStroke(BaseModel):
    """Um traço da página, como o Xournal++ o entrega/recebe."""

    x: list[float] = Field(min_length=1)
    y: list[float] = Field(min_length=1)
    color: int | None = None  # RGB (0xRRGGBB)
    width: float | None = None

    @model_validator(mode="after")
    def _same_length(self) -> "PageStroke":
        if len(self.x) != len(self.y):
            raise ValueError("x e y devem ter o mesmo tamanho")
        return self


class PageInk(BaseModel):
    """Corpo de POST /page/process: todos os traços da página/camada atual."""

    strokes: list[PageStroke]


class ExpressionResult(BaseModel):
    """Uma conta/equação encontrada: LaTeX, avaliação/classificação e a tinta do resultado."""

    latex: str
    result: str | None = None  # valor calculado (contas) ou solução (equações 1-var)
    kind: str | None = None  # tipo da curva/superfície (equações: "circunferencia"…)
    description: str | None = None  # descrição pt-BR (equações classificadas)
    error: str | None = None
    strokes: list[PageStroke]  # tinta a desenhar (vazia se não avaliável)


class PageProcessResponse(BaseModel):
    expressions: list[ExpressionResult]


class SketchGuess(BaseModel):
    label: str  # categoria QuickDraw (en)
    label_pt: str  # rótulo exibido
    confidence: float


class SketchRecognizeResponse(BaseModel):
    """POST /sketch/recognize: melhor palpite + topk + rótulo pronto em tinta."""

    label: str
    label_pt: str
    confidence: float
    topk: list[SketchGuess]
    strokes: list[PageStroke]  # tinta do rótulo, abaixo do desenho
