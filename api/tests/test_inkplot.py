"""Testes do gerador de gráficos em tinta (hmer_api.inkplot)."""

import math

import pytest
import sympy
from hmer_api.analyze import analyze
from hmer_api.inkplot import analysis_to_strokes

x, y, z = sympy.symbols("x y z")

CASES = [
    ("circunferencia", sympy.Eq(x**2 + y**2, 4)),
    ("elipse", sympy.Eq(x**2 / 4 + y**2, 1)),
    ("parabola", sympy.Eq(y, x**2)),
    ("hiperbole", sympy.Eq(x**2 - y**2, 1)),
    ("reta", sympy.Eq(y, 2 * x + 1)),
    ("funcao_2d", sympy.Eq(y, sympy.sin(x))),
    ("esfera", sympy.Eq(x**2 + y**2 + z**2, 4)),
    ("elipsoide", sympy.Eq(x**2 / 4 + y**2 / 9 + z**2, 1)),
    ("paraboloide_eliptico", sympy.Eq(z, x**2 + y**2)),
    ("paraboloide_hiperbolico", sympy.Eq(z, x**2 - y**2)),
    ("hiperboloide_1f", sympy.Eq(x**2 + y**2 - z**2, 1)),
    ("hiperboloide_2f", sympy.Eq(z**2 - x**2 - y**2, 1)),
    ("cone", sympy.Eq(x**2 + y**2, z**2)),
    ("plano", sympy.Eq(x + y + z, 1)),
    ("funcao_3d", sympy.Eq(z, sympy.sin(x) + y / 2)),
]


@pytest.mark.parametrize("kind,eq", CASES, ids=[k for k, _ in CASES])
def test_every_kind_renders_inside_frame(kind, eq):
    a = analyze(eq)
    assert a is not None and a.kind == kind
    strokes = analysis_to_strokes(a, x=100.0, y=50.0, size=200.0)
    assert len(strokes) >= 3  # eixos + curva no mínimo
    pts = [p for s in strokes for p in s]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # dentro do quadro, com folga p/ rótulos dos eixos (ancoram além da ponta e vazam)
    assert min(xs) >= 60.0 and max(xs) <= 340.0
    assert min(ys) >= 10.0 and max(ys) <= 340.0
    assert all(len(s) >= 2 for s in strokes)
    assert all(math.isfinite(c) for p in pts for c in p)


def test_circle_curve_closes():
    a = analyze(sympy.Eq(x**2 + y**2, 4))
    strokes = analysis_to_strokes(a, x=0.0, y=0.0, size=200.0)
    longest = max(strokes, key=len)
    dx = longest[0][0] - longest[-1][0]
    dy = longest[0][1] - longest[-1][1]
    assert math.hypot(dx, dy) < 2.0  # fecha


def test_scale_is_uniform():
    # a circunferência mapeada tem de continuar redonda (aspecto preservado)
    a = analyze(sympy.Eq(x**2 + y**2, 4))
    strokes = analysis_to_strokes(a, x=0.0, y=0.0, size=300.0)
    longest = max(strokes, key=len)
    xs = [p[0] for p in longest]
    ys = [p[1] for p in longest]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    assert abs(w - h) < 2.0
