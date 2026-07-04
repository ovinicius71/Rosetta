"""Testes da segmentação de contas (hmer_ml.segment) com geometria sintética."""

from hmer_ml.data.ink import Point, Stroke
from hmer_ml.segment import (
    RESULT_COLOR,
    detect_equals,
    find_pending,
    group_expression,
)


def line(x0, y0, x1, y1, n=8) -> Stroke:
    """Segmento reto amostrado em n pontos."""
    return Stroke(
        points=[
            Point(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1))
            for i in range(n)
        ]
    )


def zigzag(cx, cy, w=10.0, h=14.0) -> Stroke:
    """Rabisco compacto (símbolo genérico: dígito/letra), nada parecido com barra."""
    return Stroke(
        points=[
            Point(cx - w / 2, cy - h / 2),
            Point(cx + w / 2, cy - h / 4),
            Point(cx - w / 2, cy),
            Point(cx + w / 2, cy + h / 4),
            Point(cx - w / 2, cy + h / 2),
        ]
    )


def equals_at(x, y, w=12.0, gap=5.0) -> list[Stroke]:
    """Par de barras de um '=' com canto superior esquerdo em (x, y)."""
    return [line(x, y, x + w, y), line(x, y + gap, x + w, y + gap)]


# ---- detect_equals ----------------------------------------------------------------------


def test_detects_simple_equals():
    strokes = equals_at(100, 50)
    eqs = detect_equals(strokes)
    assert len(eqs) == 1
    assert set(eqs[0].stroke_indices) == {0, 1}


def test_retraced_dash_is_not_equals():
    # mesmo traço passado duas vezes (gap vertical ~0) = hífen reforçado, não '='
    strokes = [line(100, 50, 112, 50), line(100, 50.3, 112, 50.3)]
    assert detect_equals(strokes) == []


def test_bars_on_different_text_lines_are_not_equals():
    # dois hífens de texto, um em cada linha (gap >> largura)
    strokes = [line(100, 50, 112, 50), line(100, 90, 112, 90)]
    assert detect_equals(strokes) == []


def test_side_by_side_dashes_are_not_equals():
    # "--" no meio de texto: sem sobreposição horizontal
    strokes = [line(100, 50, 112, 50), line(120, 50, 132, 50)]
    assert detect_equals(strokes) == []


def test_not_equal_sign_is_rejected():
    # '≠' = par de barras + barra diagonal cruzando o miolo
    strokes = equals_at(100, 50) + [line(103, 62, 109, 43)]
    assert detect_equals(strokes) == []


def test_curly_stroke_is_not_a_bar():
    # rabisco com corda horizontal mas cheio de curvas não vira metade de '='
    strokes = [zigzag(100, 50, w=12, h=2), line(94, 55, 106, 55)]
    assert detect_equals(strokes) == []


# ---- group_expression / find_pending ----------------------------------------------------


def _conta_2_mais_3() -> list[Stroke]:
    """'2 + 3 =' na linha y≈50, com símbolos de ~14 de altura."""
    return [
        zigzag(20, 50),  # "2"
        line(35, 50, 45, 50),  # barra horizontal do "+"
        line(40, 44, 40, 56),  # barra vertical do "+"
        zigzag(60, 50),  # "3"
        *equals_at(75, 47),  # "="
    ]


def test_groups_whole_expression():
    strokes = _conta_2_mais_3()
    eqs = detect_equals(strokes)
    assert len(eqs) == 1
    expr = group_expression(strokes, eqs[0])
    assert expr.stroke_indices == [0, 1, 2, 3, 4, 5]


def test_ignores_other_lines_and_far_columns():
    strokes = _conta_2_mais_3()
    n_conta = len(strokes)
    strokes.append(zigzag(40, 100))  # texto em outra linha
    strokes.append(zigzag(300, 50))  # outra coluna, mesma linha (gap enorme)
    exprs = find_pending(strokes)
    assert len(exprs) == 1
    assert exprs[0].stroke_indices == list(range(n_conta))


def test_skips_equals_that_already_has_result():
    strokes = _conta_2_mais_3()
    colors = [None] * len(strokes)
    # resultado desenhado pela integração logo após o '='
    strokes.append(zigzag(100, 50))
    colors.append(RESULT_COLOR)
    assert find_pending(strokes, colors=colors) == []


def test_result_color_never_joins_expression():
    strokes = _conta_2_mais_3()
    colors = [None] * len(strokes)
    # tinta de resultado de OUTRA conta, à esquerda; não pode entrar no grupo
    strokes.insert(0, zigzag(5, 50))
    colors.insert(0, RESULT_COLOR)
    exprs = find_pending(strokes, colors=colors)
    assert len(exprs) == 1
    assert 0 not in exprs[0].stroke_indices


def test_two_expressions_on_different_lines():
    strokes = _conta_2_mais_3()
    n_first = len(strokes)
    second = [zigzag(20, 120), *equals_at(40, 117)]
    strokes += second
    exprs = find_pending(strokes)
    assert len(exprs) == 2
    assert exprs[0].stroke_indices == list(range(n_first))
    assert exprs[1].stroke_indices == list(range(n_first, len(strokes)))
