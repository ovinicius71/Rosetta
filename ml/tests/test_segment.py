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


def _equacao_com_rhs() -> list[Stroke]:
    """'<algo> <algo> = <algo>' na linha y≈50 (equação com lado direito, ex. x+y=4)."""
    return [
        zigzag(20, 50),  # "x"
        zigzag(45, 50),  # "y"
        *equals_at(60, 47),  # "="
        zigzag(95, 50),  # "4"
    ]


def test_equation_groups_both_sides():
    strokes = _equacao_com_rhs()
    exprs = find_pending(strokes)
    assert len(exprs) == 1
    e = exprs[0]
    assert e.stroke_indices == [0, 1, 2, 3, 4]
    assert e.lhs_indices == [0, 1]
    assert e.rhs_indices == [4]


def test_conta_has_empty_rhs():
    strokes = _conta_2_mais_3()
    e = find_pending(strokes)[0]
    assert e.rhs_indices == []
    assert e.lhs_indices == [0, 1, 2, 3]


def test_superscript_hovering_over_equals_is_adopted():
    """Expoente que invade o espaço do '=' por cima (ex.: o '²' de "y²" escrito alto)
    entra no LHS — as varreduras laterais sozinhas o perderiam (regressão da tinta
    real do usuário em 2026-07-04)."""
    strokes = [
        zigzag(20, 50),  # "x"
        zigzag(45, 50),  # "y"
        zigzag(64, 34, w=8.0, h=10.0),  # "²" alto, pairando sobre o início do '='
        *equals_at(60, 47),  # "="  (x 60..72)
        zigzag(95, 50),  # "4"
    ]
    e = find_pending(strokes)[0]
    assert 2 in e.lhs_indices
    assert e.lhs_indices == [0, 1, 2]
    assert e.rhs_indices == [5]


def test_adoption_ignores_text_line_above():
    """Outra linha de escrita bem acima NÃO é adotada (banda vertical limita)."""
    strokes = [
        zigzag(20, 10),  # texto de outra linha, acima da conta
        zigzag(20, 50),
        zigzag(45, 50),
        *equals_at(60, 47),
        zigzag(95, 50),
    ]
    e = find_pending(strokes)[0]
    assert 0 not in e.stroke_indices


def test_result_below_marks_equation_as_solved():
    strokes = _equacao_com_rhs()
    colors: list[int | None] = [None] * len(strokes)
    # gráfico desenhado logo abaixo da equação (tinta de resultado)
    strokes.append(zigzag(60, 95, w=60, h=40))
    colors.append(RESULT_COLOR)
    assert find_pending(strokes, colors=colors) == []


def test_result_below_other_column_does_not_mark():
    strokes = _equacao_com_rhs()
    colors: list[int | None] = [None] * len(strokes)
    # tinta de resultado abaixo mas SEM sobreposição horizontal (de outra expressão)
    strokes.append(zigzag(300, 95))
    colors.append(RESULT_COLOR)
    assert len(find_pending(strokes, colors=colors)) == 1


def test_two_expressions_on_different_lines():
    strokes = _conta_2_mais_3()
    n_first = len(strokes)
    second = [zigzag(20, 120), *equals_at(40, 117)]
    strokes += second
    exprs = find_pending(strokes)
    assert len(exprs) == 2
    assert exprs[0].stroke_indices == list(range(n_first))
    assert exprs[1].stroke_indices == list(range(n_first, len(strokes)))
