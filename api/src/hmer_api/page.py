"""Pipeline de POST /page/process: contas de uma página do Xournal++ → resultados em tinta.

Recebe todos os traços da página (plugin Lua), encontra as contas pendentes
(hmer_ml.segment), reconhece cada uma (Recognizer), avalia (SymPy) e devolve o resultado
como traços Hershey prontos para app.addStrokes — na cor RESULT_COLOR, que é também a marca
de "já processado" nas execuções seguintes.
"""

from __future__ import annotations

from fastapi import HTTPException

from hmer_ml.data.ink import Ink, Point, Stroke, normalize, resample
from hmer_ml.segment import RESULT_COLOR, Expression, find_pending

from .evaluate import evaluate_latex
from .inkfont import text_to_strokes
from .recognize import get_recognizer
from .schemas import ExpressionResult, PageInk, PageProcessResponse, PageStroke

# Posição/tamanho do resultado, em larguras-de-'=' (mesma convenção de hmer_ml.segment).
RESULT_GAP = 0.9  # espaço entre o '=' e o resultado
MIN_HEIGHT = 1.2  # altura mínima do texto do resultado
MAX_HEIGHT = 3.0  # altura máxima (frações altas não devem gerar resultado gigante)
DEFAULT_PEN_WIDTH = 1.41  # width padrão da caneta do Xournal++

# O modelo é sensível à densidade de pontos (features são deltas ponto-a-ponto e o treino
# do CROHME não reamostra): a caneta do Xournal++ amostra bem mais denso que o CROHME e
# isso sozinho derruba o reconhecimento. Antes de reconhecer, normalizamos e reamostramos
# ao passo MEDIANO do CROHME em coordenadas normalizadas — a tinta cai no regime que o
# modelo viu em treino, qualquer que seja o dispositivo.
RESAMPLE_STEP = 0.004


def process_page(page: PageInk) -> PageProcessResponse:
    """Processa a página inteira. Erros de uma conta não derrubam as demais."""
    strokes = [
        Stroke(points=[Point(x, y) for x, y in zip(ps.x, ps.y)]) for ps in page.strokes
    ]
    colors = [ps.color for ps in page.strokes]

    recognizer = get_recognizer()
    out: list[ExpressionResult] = []
    for expr in find_pending(strokes, colors=colors):
        # Só o lado esquerdo vai ao modelo: o CROHME quase não tem expressões terminando
        # em '=', então incluir os traços do '=' joga a conta para fora da distribuição.
        lhs = [i for i in expr.stroke_indices if i not in expr.equals.stroke_indices]
        if not lhs:
            continue  # '=' sem nada à esquerda não é uma conta
        ink = resample(
            normalize(Ink(strokes=[strokes[i] for i in lhs])), RESAMPLE_STEP
        )
        try:
            latex = recognizer.recognize(ink.to_dict())
        except HTTPException as e:
            if e.status_code == 501:
                raise  # sem modelo não há o que processar — falha da requisição toda
            out.append(ExpressionResult(latex="", error=str(e.detail), strokes=[]))
            continue

        evaluation = evaluate_latex(latex.rstrip().removesuffix("="))
        if evaluation.result is None:
            out.append(ExpressionResult(latex=latex, error=evaluation.error, strokes=[]))
            continue

        out.append(
            ExpressionResult(
                latex=latex,
                result=evaluation.result,
                strokes=_result_strokes(evaluation.result, expr, page),
            )
        )
    return PageProcessResponse(expressions=out)


def _result_strokes(text: str, expr: Expression, page: PageInk) -> list[PageStroke]:
    """Tinta do resultado: logo à direita do '=', na altura da linha de escrita."""
    eq = expr.equals
    eq_w = max(eq.width, 1e-9)
    x = eq.bbox[2] + RESULT_GAP * eq_w
    y_mid = (eq.bbox[1] + eq.bbox[3]) / 2.0
    line_height = expr.bbox[3] - expr.bbox[1]
    height = max(MIN_HEIGHT * eq_w, min(line_height, MAX_HEIGHT * eq_w))
    pen = _pen_width(expr, page)

    return [
        PageStroke(
            x=[p[0] for p in poly],
            y=[p[1] for p in poly],
            color=RESULT_COLOR,
            width=pen,
        )
        for poly in text_to_strokes(text, x=x, y_mid=y_mid, height=height)
    ]


def _pen_width(expr: Expression, page: PageInk) -> float:
    """Casa a espessura do resultado com a caneta usada na conta."""
    widths = sorted(
        page.strokes[i].width
        for i in expr.stroke_indices
        if page.strokes[i].width is not None
    )
    if not widths:
        return DEFAULT_PEN_WIDTH
    return widths[len(widths) // 2]
