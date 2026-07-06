"""Pipeline de POST /page/process: contas de uma página do Xournal++ → resultados em tinta.

Recebe todos os traços da página (plugin Lua), encontra as contas pendentes
(hmer_ml.segment), reconhece cada uma (Recognizer), avalia (SymPy) e devolve o resultado
como traços Hershey prontos para app.addStrokes — na cor RESULT_COLOR, que é também a marca
de "já processado" nas execuções seguintes.
"""

from __future__ import annotations

from fastapi import HTTPException

from hmer_ml.data.ink import Ink, Point, Stroke
from hmer_ml.segment import RESULT_COLOR, Expression, find_pending

from .analyze import Analysis, analyze
from .evaluate import evaluate_latex
from .inkfont import strip_accents, text_to_strokes, text_width
from .inkplot import analysis_to_strokes
from .recognize import get_recognizer
from .schemas import ExpressionResult, PageInk, PageProcessResponse, PageStroke

# Posição/tamanho do resultado, em larguras-de-'=' (mesma convenção de hmer_ml.segment).
RESULT_GAP = 0.9  # espaço entre o '=' e o resultado
MIN_HEIGHT = 1.2  # altura mínima do texto do resultado
MAX_HEIGHT = 3.0  # altura máxima (frações altas não devem gerar resultado gigante)
DEFAULT_PEN_WIDTH = 1.41  # width padrão da caneta do Xournal++

# Gráficos de equações (abaixo da escrita)
PLOT_GAP = 0.8  # espaço entre a equação e o gráfico / altura da linha de escrita
PLOT_SIZE_FACTOR = 4.0  # largura do gráfico / altura da linha de escrita
PLOT_SIZE_MIN = 150.0
PLOT_SIZE_MAX = 340.0
DESC_GAP = 0.8  # espaço entre o gráfico e a descrição / altura do texto da descrição


def process_page(page: PageInk) -> PageProcessResponse:
    """Processa a página inteira. Erros de uma conta não derrubam as demais."""
    strokes = [
        Stroke(points=[Point(x, y) for x, y in zip(ps.x, ps.y)]) for ps in page.strokes
    ]
    colors = [ps.color for ps in page.strokes]

    recognizer = get_recognizer()
    out: list[ExpressionResult] = []
    for expr in find_pending(strokes, colors=colors):
        if not expr.lhs_indices:
            continue  # '=' sem nada à esquerda não é uma conta nem equação
        is_equation = bool(expr.rhs_indices)
        # Conta ("2+3="): só o LHS vai ao modelo — o CROHME quase não tem expressões
        # TERMINANDO em '=' (OOD). Equação ("x²+y²=4"): vai completa, com o '=' no meio
        # (essas estão na distribuição do CROHME).
        send = expr.stroke_indices if is_equation else expr.lhs_indices
        # A canonicalização (normalize + resample do config) é do Recognizer — o mesmo
        # caminho do treino. Reamostrar aqui de novo distorceria a tinta.
        ink = Ink(strokes=[strokes[i] for i in send])
        try:
            if is_equation:
                # equação: todas as hipóteses do beam — a 1ª leitura CLASSIFICÁVEL
                # (cônica/quádrica/função) vence; leituras trocadas de 1 símbolo
                # raramente são todas implotáveis.
                topk = getattr(recognizer, "recognize_topk", None)
                latexes = topk(ink.to_dict()) if topk else [recognizer.recognize(ink.to_dict())]
            else:
                latexes = [recognizer.recognize(ink.to_dict())]
        except HTTPException as e:
            if e.status_code == 501:
                raise  # sem modelo não há o que processar — falha da requisição toda
            out.append(ExpressionResult(latex="", error=str(e.detail), strokes=[]))
            continue

        out.append(
            _equation_result(latexes, expr, page)
            if is_equation
            else _conta_result(latexes[0], expr, page)
        )
    return PageProcessResponse(expressions=out)


def _conta_result(latex: str, expr: Expression, page: PageInk) -> ExpressionResult:
    """Conta: avalia o LHS e escreve o valor logo após o '='."""
    evaluation = evaluate_latex(latex.rstrip().removesuffix("="))
    if evaluation.result is None:
        return ExpressionResult(latex=latex, error=evaluation.error, strokes=[])
    return ExpressionResult(
        latex=latex,
        result=evaluation.result,
        strokes=_result_strokes(evaluation.result, expr, page),
    )


def _equation_result(
    latexes: list[str], expr: Expression, page: PageInk
) -> ExpressionResult:
    """Equação: classifica (cônica/quádrica/função) e desenha o gráfico com eixos
    abaixo da escrita; sem classificação, cai no solve e escreve a solução abaixo
    (a direita do '=' está ocupada pelo próprio RHS).

    Recebe as hipóteses do beam em ordem de confiança: a 1ª que classificar vence."""
    from sympy.parsing.latex import parse_latex

    latex = latexes[0]
    analysis = None
    for cand in latexes:
        try:
            analysis = analyze(parse_latex(cand))
        except Exception:  # noqa: BLE001 - LaTeX não parseável → próxima hipótese
            continue
        if analysis is not None:
            latex = cand
            break

    if analysis is not None:
        return ExpressionResult(
            latex=latex,
            kind=analysis.kind,
            description=analysis.description_pt,
            strokes=_plot_strokes(analysis, expr, page),
        )

    evaluation = evaluate_latex(latex)
    if evaluation.result is None:
        return ExpressionResult(latex=latex, error=evaluation.error, strokes=[])
    return ExpressionResult(
        latex=latex,
        result=evaluation.result,
        strokes=_below_text_strokes(evaluation.result, expr, page),
    )


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


def _plot_strokes(analysis: Analysis, expr: Expression, page: PageInk) -> list[PageStroke]:
    """Gráfico com eixos abaixo da equação + descrição em Hershey sob o gráfico."""
    x0, _, x1, y1 = expr.bbox
    line_h = max(y1 - expr.bbox[1], 1e-6)
    size = max(PLOT_SIZE_MIN, min(PLOT_SIZE_FACTOR * line_h, PLOT_SIZE_MAX))
    px = (x0 + x1) / 2.0 - size / 2.0
    py = y1 + PLOT_GAP * line_h
    pen = _pen_width(expr, page)

    polylines = analysis_to_strokes(analysis, x=px, y=py, size=size)
    bottom = max(p[1] for poly in polylines for p in poly)
    desc = strip_accents(analysis.description_pt)
    desc_h = max(11.0, min(0.05 * size, 14.0))
    desc_w = text_width(desc, desc_h)
    polylines += text_to_strokes(
        desc,
        x=px + size / 2.0 - desc_w / 2.0,
        y_mid=bottom + DESC_GAP * desc_h + desc_h / 2.0,
        height=desc_h,
    )
    return [
        PageStroke(x=[p[0] for p in poly], y=[p[1] for p in poly],
                   color=RESULT_COLOR, width=pen)
        for poly in polylines
    ]


def _below_text_strokes(text: str, expr: Expression, page: PageInk) -> list[PageStroke]:
    """Solução escrita ABAIXO da expressão (equações: a direita do '=' está ocupada)."""
    x0, _, x1, y1 = expr.bbox
    eq_w = max(expr.equals.width, 1e-9)
    line_h = max(y1 - expr.bbox[1], 1e-6)
    height = max(MIN_HEIGHT * eq_w, min(line_h, MAX_HEIGHT * eq_w))
    # teto de largura: soluções longas (ex.: "X = -sqrt(8 - t), X = sqrt(8 - t)") não
    # podem sair gigantes — encolhe a altura até caber em ~1.5x a largura da expressão
    max_w = max(1.5 * (x1 - x0), PLOT_SIZE_MIN)
    w = text_width(text, height)
    if w > max_w:
        height = max(10.0, height * max_w / w)
    pen = _pen_width(expr, page)
    x = (x0 + x1) / 2.0 - text_width(text, height) / 2.0
    y_mid = y1 + PLOT_GAP * line_h + height / 2.0
    return [
        PageStroke(x=[p[0] for p in poly], y=[p[1] for p in poly],
                   color=RESULT_COLOR, width=pen)
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
