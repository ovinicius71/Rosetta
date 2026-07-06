"""Segmentação de contas numa página de anotações livres (integração Xournal++).

Dado o conjunto de traços de uma página (matemática misturada com texto/desenhos), encontra
os sinais de "=" por geometria e agrupa os traços que formam cada conta — a área é
"selecionada" automaticamente, sem laço. Consumido por POST /page/process na API.

Depende só da stdlib e opera sobre hmer_ml.data.ink.Stroke (mesmo contrato do modelo).
Convenção de eixos: y cresce para baixo (canvas web e página do Xournal++).

Heurísticas calibráveis nas constantes abaixo; unidades são relativas ao próprio "="
(invariante a escala da escrita).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .data.ink import Stroke

# ---- calibração -------------------------------------------------------------------------
# Um "traço-barra" (candidato a metade de um "="):
MAX_BAR_ANGLE_DEG = 25.0  # inclinação máxima da corda em relação à horizontal
MAX_BAR_DEVIATION = 0.20  # desvio perpendicular máx. dos pontos / comprimento da corda
MAX_BAR_WINDING = 1.35  # comprimento de arco / corda (rejeita curvas fechadas "retas")

# Um "=" (par de barras):
MIN_PAIR_LENGTH_RATIO = 0.5  # corda menor / corda maior
MIN_PAIR_H_OVERLAP = 0.6  # sobreposição horizontal / largura da barra menor
MIN_PAIR_V_GAP = 0.08  # gap vertical entre os centros / largura média (0 = retraçado)
MAX_PAIR_V_GAP = 1.0  # acima disso são duas linhas de texto, não um "="
MAX_CROSSING_OVERLAP = 0.35  # traço alheio cobrindo o miolo do "=" (rejeita "≠")

# Agrupamento da conta (à esquerda do "="):
MAX_SYMBOL_GAP = 2.5  # gap horizontal máx. entre símbolos vizinhos / largura do "="
BAND_SLACK = 0.75  # folga vertical da banda da linha de escrita / largura do "="
EQ_X_TOLERANCE = 0.25  # quanto um símbolo pode invadir o "=" pela direita / largura do "="

# Resultado desenhado pela integração (cor fixa: é assim que marcamos "já processado").
RESULT_COLOR = 0xE8590C  # laranja
RESULT_SEARCH_SPAN = 6.0  # até onde procurar um resultado à direita do "=" / largura do "="
RESULT_BELOW_SPAN = 5.0  # até onde procurar um resultado ABAIXO da expressão (gráficos)

BBox = tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


@dataclass
class EqualsSign:
    """Par de barras identificado como sinal de igual."""

    stroke_indices: tuple[int, int]
    bbox: BBox

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass
class Expression:
    """Uma conta/equação: os traços da expressão (incluindo o próprio '=').

    `lhs_indices`/`rhs_indices` separam os lados do '=' — RHS vazio é uma conta
    ("2+3="); RHS não-vazio é uma equação ("x²+y²=4").
    """

    stroke_indices: list[int]
    equals: EqualsSign
    bbox: BBox
    lhs_indices: list[int] = field(default_factory=list)
    rhs_indices: list[int] = field(default_factory=list)


def _stroke_bbox(s: Stroke) -> BBox:
    xs = [p.x for p in s.points]
    ys = [p.y for p in s.points]
    return min(xs), min(ys), max(xs), max(ys)


def _merge(a: BBox, b: BBox) -> BBox:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _arc_length(s: Stroke) -> float:
    return sum(
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(s.points, s.points[1:])
    )


def _is_bar(s: Stroke) -> bool:
    """Traço quase reto e quase horizontal (candidato a metade de um '=')."""
    if len(s.points) < 2:
        return False
    p0, p1 = s.points[0], s.points[-1]
    dx, dy = p1.x - p0.x, p1.y - p0.y
    chord = math.hypot(dx, dy)
    if chord <= 0:
        return False
    if abs(math.degrees(math.atan2(abs(dy), abs(dx)))) > MAX_BAR_ANGLE_DEG:
        return False
    # desvio perpendicular máximo dos pontos em relação à corda
    max_dev = max(
        abs((p.x - p0.x) * dy - (p.y - p0.y) * dx) / chord for p in s.points
    )
    if max_dev > MAX_BAR_DEVIATION * chord:
        return False
    # curvas que voltam sobre si (círculos achatados) têm arco >> corda
    return _arc_length(s) / chord <= MAX_BAR_WINDING


def _h_overlap(a: BBox, b: BBox) -> float:
    """Sobreposição horizontal relativa à barra mais estreita."""
    inter = min(a[2], b[2]) - max(a[0], b[0])
    narrow = min(a[2] - a[0], b[2] - b[0])
    if narrow <= 0:
        return 0.0
    return inter / narrow


def detect_equals(strokes: list[Stroke], candidates: list[int] | None = None) -> list[EqualsSign]:
    """Encontra sinais de '=' entre os traços (pares de barras empilhadas).

    `candidates` restringe a busca a esses índices (ex.: excluir traços de resultado).
    Cada traço participa de no máximo um '='; pares mais próximos verticalmente vencem.
    """
    idxs = list(range(len(strokes))) if candidates is None else list(candidates)
    bars = [i for i in idxs if _is_bar(strokes[i])]
    boxes = {i: _stroke_bbox(strokes[i]) for i in idxs}

    scored: list[tuple[float, int, int]] = []
    for a_pos, i in enumerate(bars):
        for j in bars[a_pos + 1 :]:
            bi, bj = boxes[i], boxes[j]
            wi, wj = bi[2] - bi[0], bj[2] - bj[0]
            if min(wi, wj) / max(wi, wj) < MIN_PAIR_LENGTH_RATIO:
                continue
            if _h_overlap(bi, bj) < MIN_PAIR_H_OVERLAP:
                continue
            mean_w = (wi + wj) / 2.0
            cy_i, cy_j = (bi[1] + bi[3]) / 2.0, (bj[1] + bj[3]) / 2.0
            v_gap = abs(cy_i - cy_j)
            if not (MIN_PAIR_V_GAP * mean_w <= v_gap <= MAX_PAIR_V_GAP * mean_w):
                continue
            scored.append((v_gap / mean_w, i, j))

    scored.sort()
    used: set[int] = set()
    out: list[EqualsSign] = []
    for _, i, j in scored:
        if i in used or j in used:
            continue
        eq_bbox = _merge(boxes[i], boxes[j])
        if _has_crossing_stroke(eq_bbox, boxes, exclude={i, j}):
            continue  # provavelmente "≠" (ou rabisco por cima)
        used.update((i, j))
        out.append(EqualsSign(stroke_indices=(i, j), bbox=eq_bbox))
    # ordem de leitura: de cima para baixo, depois esquerda → direita
    out.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    return out


def _has_crossing_stroke(eq_bbox: BBox, boxes: dict[int, BBox], exclude: set[int]) -> bool:
    """True se algum outro traço cobre boa parte do miolo do '=' (caso do '≠')."""
    ex0, ey0, ex1, ey1 = eq_bbox
    area = max((ex1 - ex0) * (ey1 - ey0), 1e-9)
    for k, b in boxes.items():
        if k in exclude:
            continue
        ix = max(0.0, min(ex1, b[2]) - max(ex0, b[0]))
        iy = max(0.0, min(ey1, b[3]) - max(ey0, b[1]))
        if (ix * iy) / area > MAX_CROSSING_OVERLAP:
            return True
    return False


def group_expression(
    strokes: list[Stroke],
    equals: EqualsSign,
    candidates: list[int] | None = None,
) -> Expression:
    """Agrupa os traços da conta/equação em torno do `equals` dado.

    Duas varreduras encadeando símbolos (esquerda→LHS e direita→RHS): um traço entra se
    estiver na banda vertical da linha de escrita (que cresce com o grupo — frações e
    expoentes cabem) e a menos de MAX_SYMBOL_GAP larguras-de-'=' do grupo.
    """
    idxs = list(range(len(strokes))) if candidates is None else list(candidates)
    eq_w = max(equals.width, 1e-9)
    eq_set = set(equals.stroke_indices)
    boxes = {i: _stroke_bbox(strokes[i]) for i in idxs if i not in eq_set}
    slack = BAND_SLACK * eq_w

    def sweep(leftward: bool) -> tuple[set[int], BBox]:
        if leftward:
            pool = [(i, b) for i, b in boxes.items() if b[2] <= equals.bbox[0] + EQ_X_TOLERANCE * eq_w]
            pool.sort(key=lambda ib: -ib[1][2])  # borda direita decrescente
        else:
            pool = [(i, b) for i, b in boxes.items() if b[0] >= equals.bbox[2] - EQ_X_TOLERANCE * eq_w]
            pool.sort(key=lambda ib: ib[1][0])  # borda esquerda crescente

        side: set[int] = set()
        bbox = equals.bbox
        for i, b in pool:
            gap = (bbox[0] - b[2]) if leftward else (b[0] - bbox[2])
            if gap > MAX_SYMBOL_GAP * eq_w:
                break  # buraco grande: o que vier depois é outra coluna/palavra
            if b[3] < bbox[1] - slack or b[1] > bbox[3] + slack:
                continue  # fora da linha de escrita (outra linha da página)
            side.add(i)
            bbox = _merge(bbox, b)
        return side, bbox

    lhs, lhs_bbox = sweep(leftward=True)
    rhs, rhs_bbox = sweep(leftward=False)
    bbox = _merge(lhs_bbox, rhs_bbox)

    # Adoção de sobrescritos órfãos: as varreduras exigem o traço INTEIRO de um lado
    # do '=', então um expoente que paira sobre o '=' (ex.: o '²' de "y²" invadindo o
    # início do "=") não entra em nenhuma. Adota traços que sobrepõem a expressão
    # horizontalmente e estão na banda vertical dela, no lado do seu centro.
    eq_cx = (equals.bbox[0] + equals.bbox[2]) / 2.0
    grouped = eq_set | lhs | rhs
    changed = True
    while changed:
        changed = False
        for i, b in boxes.items():
            if i in grouped:
                continue
            overlaps_h = min(bbox[2], b[2]) - max(bbox[0], b[0]) > 0
            in_band = not (b[3] < bbox[1] - slack or b[1] > bbox[3] + slack)
            if not (overlaps_h and in_band):
                continue
            (lhs if (b[0] + b[2]) / 2.0 < eq_cx else rhs).add(i)
            grouped.add(i)
            bbox = _merge(bbox, b)
            changed = True

    return Expression(
        stroke_indices=sorted(eq_set | lhs | rhs),
        equals=equals,
        bbox=bbox,
        lhs_indices=sorted(lhs),
        rhs_indices=sorted(rhs),
    )


def _has_result(expr: Expression, result_boxes: list[BBox]) -> bool:
    """True se já existe tinta de resultado desta expressão.

    Dois lugares possíveis: logo à **direita** do '=' na mesma linha (contas: "2+3= 5")
    ou logo **abaixo** do bbox da expressão (equações: gráfico/solução desenhados sob
    "x²+y²=4").
    """
    ex0, ey0, ex1, ey1 = expr.equals.bbox
    eq_w = max(expr.equals.width, 1e-9)
    slack = BAND_SLACK * eq_w
    bx0, by0, bx1, by1 = expr.bbox
    for b in result_boxes:
        # caso conta: à direita do '=', na banda da linha de escrita
        right_of_eq = ex1 - EQ_X_TOLERANCE * eq_w <= b[0] <= ex1 + RESULT_SEARCH_SPAN * eq_w
        same_line = not (b[3] < ey0 - slack or b[1] > ey1 + slack)
        if right_of_eq and same_line:
            return True
        # caso equação: abaixo da expressão, com sobreposição horizontal
        below = by1 - slack <= b[1] <= by1 + RESULT_BELOW_SPAN * eq_w
        overlaps_h = min(bx1, b[2]) - max(bx0, b[0]) > 0
        if below and overlaps_h:
            return True
    return False


def find_pending(
    strokes: list[Stroke],
    colors: list[int | None] | None = None,
    result_color: int = RESULT_COLOR,
) -> list[Expression]:
    """Pipeline completo: contas ainda sem resultado desenhado, em ordem de leitura.

    `colors[i]` é a cor do traço i (paralela a `strokes`); traços na cor de resultado são
    marcas de execuções anteriores — nunca entram numa conta e indicam '=' já resolvido.
    """
    if colors is None:
        colors = [None] * len(strokes)
    user_idx = [i for i in range(len(strokes)) if colors[i] != result_color]
    result_boxes = [
        _stroke_bbox(strokes[i]) for i in range(len(strokes)) if colors[i] == result_color
    ]

    out: list[Expression] = []
    claimed: set[int] = set()
    for eq in detect_equals(strokes, candidates=user_idx):
        # agrupa primeiro (o teste de "já resolvida" precisa do bbox completo p/ olhar
        # abaixo da expressão); expressões resolvidas ainda reivindicam seus traços.
        free = [i for i in user_idx if i not in claimed or i in eq.stroke_indices]
        expr = group_expression(strokes, eq, candidates=free)
        claimed.update(expr.stroke_indices)
        if _has_result(expr, result_boxes):
            continue
        out.append(expr)
    return out
