"""Testes do classificador de equações (hmer_api.analyze)."""

import pytest
import sympy
from hmer_api.analyze import analyze

x, y, z, w = sympy.symbols("x y z w")
X, Y = sympy.symbols("X Y")


def _eq(lhs, rhs=0):
    return sympy.Eq(lhs, rhs)


# ── R² ─────────────────────────────────────────────────────────────────────────────────


def test_circle_canonical():
    a = analyze(_eq(x**2 + y**2, 4))
    assert a is not None
    assert a.kind == "circunferencia"
    assert "raio 2" in a.description_pt
    assert "centro (0, 0)" in a.description_pt
    assert a.params["axes"] == pytest.approx([2.0, 2.0])


def test_circle_uppercase_vars():
    a = analyze(_eq(X**2 + Y**2, 4))  # o modelo pode reconhecer maiúsculas
    assert a is not None and a.kind == "circunferencia"


def test_translated_circle():
    a = analyze(_eq((x - 1) ** 2 + (y + 2) ** 2, 9))
    assert a.kind == "circunferencia"
    assert "raio 3" in a.description_pt
    assert a.params["center"] == pytest.approx([1.0, -2.0])


def test_ellipse():
    a = analyze(_eq(x**2 / 4 + y**2, 1))
    assert a.kind == "elipse"
    assert sorted(a.params["axes"]) == pytest.approx([1.0, 2.0])


def test_hyperbola():
    a = analyze(_eq(x**2 - y**2, 1))
    assert a.kind == "hiperbole"
    assert a.params["axes"] == pytest.approx([1.0, 1.0])


def test_parabola():
    a = analyze(_eq(y, x**2))
    assert a.kind == "parabola"
    assert "vertice (0, 0)" in a.description_pt


def test_line():
    a = analyze(_eq(y, 2 * x + 1))
    assert a.kind == "reta"
    assert "y = 2x + 1" in a.description_pt


def test_function_graph_2d():
    a = analyze(_eq(y, sympy.sin(x)))
    assert a.kind == "funcao_2d"
    assert a.params["expr"] == "sin(x)"


def test_empty_conic_is_none():
    assert analyze(_eq(x**2 + y**2, -1)) is None


def test_single_variable_is_none():
    assert analyze(_eq(2 * x + 4, 10)) is None


def test_any_variable_pair_is_a_conic():
    """Qualquer par de variáveis vale: w²+y²=4 é circunferência no plano y-w.

    (u,v/a,b são legítimos em cadernos — e o reconhecedor troca letras; a leitura
    estruturalmente certa ainda rende um gráfico útil.)"""
    a = analyze(_eq(w**2 + y**2, 4))
    assert a is not None and a.kind == "circunferencia"
    assert a.params["vars"] == ("y", "w")  # y tem prioridade de eixo horizontal


def test_two_vars_beyond_xy_make_parabola():
    """x²+t=8 (leitura comum p/ 'x²+y²=4' mal reconhecido) → parábola no plano x-t."""
    t = sympy.Symbol("t")
    a = analyze(_eq(x**2 + t, 8))
    assert a is not None and a.kind == "parabola"
    assert a.params["vars"] == ("x", "t")


# ── R³ ─────────────────────────────────────────────────────────────────────────────────


def test_sphere():
    a = analyze(_eq(x**2 + y**2 + z**2, 4))
    assert a.kind == "esfera"
    assert "raio 2" in a.description_pt
    assert "centro (0, 0, 0)" in a.description_pt


def test_ellipsoid():
    a = analyze(_eq(x**2 / 4 + y**2 / 9 + z**2, 1))
    assert a.kind == "elipsoide"


def test_elliptic_paraboloid():
    a = analyze(_eq(z, x**2 + y**2))
    assert a.kind == "paraboloide_eliptico"
    assert "vertice (0, 0, 0)" in a.description_pt


def test_hyperbolic_paraboloid():
    a = analyze(_eq(z, x**2 - y**2))
    assert a.kind == "paraboloide_hiperbolico"


def test_hyperboloid_one_sheet():
    a = analyze(_eq(x**2 + y**2 - z**2, 1))
    assert a.kind == "hiperboloide_1f"


def test_hyperboloid_two_sheets():
    a = analyze(_eq(z**2 - x**2 - y**2, 1))
    assert a.kind == "hiperboloide_2f"


def test_cone():
    a = analyze(_eq(x**2 + y**2, z**2))
    assert a.kind == "cone"


def test_plane():
    a = analyze(_eq(x + y + z, 1))
    assert a.kind == "plano"


def test_function_graph_3d():
    a = analyze(_eq(z, sympy.sin(x) + y))
    assert a.kind == "funcao_3d"


def test_empty_quadric_is_none():
    assert analyze(_eq(x**2 + y**2 + z**2, -1)) is None
