"""Testes da fonte de traços (hmer_api.inkfont)."""

import pytest
from hmer_api.inkfont import text_to_strokes, text_width


def _bbox(strokes):
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    return min(xs), min(ys), max(xs), max(ys)


def test_digits_have_requested_height_and_position():
    strokes = text_to_strokes("0", x=100.0, y_mid=50.0, height=20.0)
    assert strokes  # '0' tem tinta
    x0, y0, x1, y1 = _bbox(strokes)
    assert y1 - y0 == pytest.approx(20.0, abs=0.5)  # altura pedida
    assert (y0 + y1) / 2 == pytest.approx(50.0, abs=0.5)  # centrado em y_mid
    assert x0 >= 100.0  # começa a partir de x

    # y cresce para baixo: o topo do dígito tem y menor que a base
    assert y0 < 50.0 < y1


def test_text_advances_left_to_right():
    one = text_to_strokes("1", x=0.0, y_mid=0.0, height=20.0)
    two = text_to_strokes("12", x=0.0, y_mid=0.0, height=20.0)
    assert _bbox(two)[2] > _bbox(one)[2]
    assert text_width("12", 20.0) > text_width("1", 20.0)


def test_typical_sympy_outputs_are_drawable():
    for txt in ("5", "3/4 = 0.75", "x = 2", "sqrt(2)", "-1.5"):
        strokes = text_to_strokes(txt, x=0.0, y_mid=0.0, height=15.0)
        assert strokes
        assert all(len(s) >= 2 for s in strokes)  # polilinhas desenháveis


def test_unknown_char_falls_back_to_question_mark():
    fallback = text_to_strokes("π", x=0.0, y_mid=0.0, height=15.0)
    question = text_to_strokes("?", x=0.0, y_mid=0.0, height=15.0)
    assert fallback == question
