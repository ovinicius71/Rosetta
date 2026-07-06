"""Classificação de equações em R²/R³ para o caderno: "x²+y²=4" → circunferência r=2.

Recebe a expressão SymPy (vinda de parse_latex no /page/process), identifica a curva
(cônicas em R², quádricas em R³, gráficos de função) e devolve um `Analysis` com a
descrição em pt-BR e os parâmetros geométricos que o inkplot usa para desenhar.

Método clássico: forma quadrática em matriz + autovalores/autovetores (numpy) — cobre
formas rotacionadas. Casos degenerados/vazios devolvem None (o pipeline cai no solve).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy

EPS = 1e-9

X, Y, Z = sympy.symbols("x y z")
_CANON = {"x": X, "y": Y, "z": Z}


@dataclass
class Analysis:
    kind: str  # ver _DESC / inkplot
    description_pt: str
    dim: int  # 2 | 3
    params: dict = field(default_factory=dict)


def _fmt(v: float) -> str:
    """Número curto para descrições: 2.0 → '2', 1.4142 → '1.41'."""
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _normalize_vars(f: sympy.Expr) -> sympy.Expr:
    """Unifica a caixa de x/y/z (X→x etc.); outras variáveis ficam como estão."""
    sub = {}
    for s in f.free_symbols:
        name = s.name.lower()
        if name in _CANON and s.name != name:
            sub[s] = _CANON[name]
    return f.subs(sub) if sub else f


def _axis_order(s: sympy.Symbol) -> tuple[int, str]:
    """x, y, z primeiro (nessa ordem: convenção de eixos), depois alfabético."""
    try:
        return (("x", "y", "z").index(s.name), s.name)
    except ValueError:
        return (3, s.name)


def analyze(expr: sympy.Basic) -> Analysis | None:
    """Equação/expressão → Analysis, ou None se não for uma curva/superfície plotável.

    Qualquer par/trio de variáveis vale ("u²+v²=1" é uma circunferência no plano u-v;
    o modelo de escrita também confunde letras, e a leitura ainda é um gráfico útil).
    """
    f = expr.lhs - expr.rhs if isinstance(expr, sympy.Equality) else expr
    f = _normalize_vars(sympy.expand(f))
    free = sorted(f.free_symbols, key=_axis_order)
    # só letras latinas viram eixo: '\gamma' etc. em geral é erro de leitura do
    # reconhecedor, e plotar "no plano x-γ" confunde mais do que ajuda
    if any(not (len(s.name) == 1 and s.name.isascii() and s.name.isalpha()) for s in free):
        return None
    if len(free) == 2:
        return _analyze_2d(f, free)
    if len(free) == 3:
        # o R³ trabalha nos símbolos canônicos; renomeia mantendo a ordem dos eixos
        f3 = f.subs(dict(zip(free, (X, Y, Z))), simultaneous=True)
        return _analyze_3d(f3)
    return None


# ── R² ─────────────────────────────────────────────────────────────────────────────────


def _analyze_2d(f: sympy.Expr, free: list[sympy.Symbol]) -> Analysis | None:
    h, v = free  # eixo horizontal e vertical (ordem alfabética: x,y / x,z / y,z)
    try:
        poly = sympy.Poly(f, h, v)
        degree = poly.total_degree()
    except sympy.PolynomialError:
        return _function_2d(f, h, v)
    if degree == 1:
        return _line(poly, h, v)
    if degree != 2:
        return _function_2d(f, h, v)

    def c(i: int, j: int) -> float:
        return float(poly.coeff_monomial(h**i * v**j) or 0)

    A, B, C = c(2, 0), c(1, 1), c(0, 2)
    D, E, F = c(1, 0), c(0, 1), c(0, 0)
    Q = np.array([[A, B / 2], [B / 2, C]])
    b = np.array([D, E])
    lam, R = np.linalg.eigh(Q)  # autovalores crescentes; R ortogonal (colunas)

    scale = max(abs(A), abs(B), abs(C), EPS)
    if abs(np.linalg.det(Q)) > EPS * scale**2:
        # cônica com centro (elipse/circunferência/hipérbole)
        center = np.linalg.solve(2 * Q, -b)
        f0 = float(f.subs({h: center[0], v: center[1]}))
        if abs(f0) < EPS * scale:
            return None  # ponto ou par de retas
        # semi-eixo ao longo de cada autovetor: λ_i t² = -f0
        ratios = [-f0 / li for li in lam]
        angle = float(np.arctan2(R[1, 1], R[0, 1]))  # direção do 2º autovetor
        if all(r > 0 for r in ratios):  # elipse real
            a1, a2 = np.sqrt(ratios[1]), np.sqrt(ratios[0])  # (maior autovalor último)
            cen = f"({_fmt(center[0])}, {_fmt(center[1])})"
            if abs(lam[0] - lam[1]) < 1e-6 * scale:
                return Analysis(
                    "circunferencia",
                    f"circunferencia · raio {_fmt(a1)} · centro {cen}",
                    2,
                    {"center": center.tolist(), "axes": [a1, a1], "angle": 0.0, "vars": (h.name, v.name)},
                )
            return Analysis(
                "elipse",
                f"elipse · semi-eixos {_fmt(a1)} e {_fmt(a2)} · centro {cen}",
                2,
                {"center": center.tolist(), "axes": [a1, a2], "angle": angle, "vars": (h.name, v.name)},
            )
        if all(r < 0 for r in ratios):
            return None  # elipse imaginária (ex.: x²+y²=-1)
        # hipérbole: eixo transverso na direção do autovetor com razão positiva
        i_pos = 0 if ratios[0] > 0 else 1
        a_t = float(np.sqrt(ratios[i_pos]))
        b_t = float(np.sqrt(-ratios[1 - i_pos]))
        t_dir = R[:, i_pos]
        angle_t = float(np.arctan2(t_dir[1], t_dir[0]))
        cen = f"({_fmt(center[0])}, {_fmt(center[1])})"
        return Analysis(
            "hiperbole",
            f"hiperbole · a={_fmt(a_t)}, b={_fmt(b_t)} · centro {cen}",
            2,
            {"center": center.tolist(), "axes": [a_t, b_t], "angle": angle_t, "vars": (h.name, v.name)},
        )

    # det ≈ 0 → parábola: no referencial dos autovetores, λ u² + d u + e w + f = 0
    i_nz = int(np.argmax(np.abs(lam)))  # autovalor não-nulo
    lam_nz = float(lam[i_nz])
    if abs(lam_nz) < EPS * scale:
        return None  # forma quadrática nula (não deveria: degree==2)
    u_dir, w_dir = R[:, i_nz], R[:, 1 - i_nz]
    d = float(b @ u_dir)
    e = float(b @ w_dir)
    if abs(e) < EPS * scale:
        return None  # degenerada (retas paralelas)
    # w = -(λu² + du + F)/e ; vértice em u* = -d/(2λ)
    u0 = -d / (2 * lam_nz)
    w0 = -(lam_nz * u0**2 + d * u0 + F) / e
    vertex = u0 * u_dir + w0 * w_dir
    k = -lam_nz / e  # w - w0 = k (u - u0)²
    vx = f"({_fmt(vertex[0])}, {_fmt(vertex[1])})"
    return Analysis(
        "parabola",
        f"parabola · vertice {vx}",
        2,
        {
            "vertex": vertex.tolist(),
            "u_dir": u_dir.tolist(),
            "w_dir": w_dir.tolist(),
            "k": k,
            "vars": (h.name, v.name),
        },
    )


def _line(poly: sympy.Poly, h: sympy.Symbol, v: sympy.Symbol) -> Analysis | None:
    a = float(poly.coeff_monomial(h) or 0)
    b = float(poly.coeff_monomial(v) or 0)
    c0 = float(poly.coeff_monomial(1) or 0)
    if abs(a) < EPS and abs(b) < EPS:
        return None
    if abs(b) > EPS:
        m, q = -a / b, -c0 / b
        desc = f"reta · {v.name} = {_fmt(m)}{h.name} {'+' if q >= 0 else '-'} {_fmt(abs(q))}" if abs(q) > EPS else f"reta · {v.name} = {_fmt(m)}{h.name}"
    else:
        desc = f"reta vertical · {h.name} = {_fmt(-c0 / a)}"
    return Analysis("reta", desc, 2, {"a": a, "b": b, "c": c0, "vars": (h.name, v.name)})


def _function_2d(f: sympy.Expr, h: sympy.Symbol, v: sympy.Symbol) -> Analysis | None:
    """v = g(h) explícita (grau 1 em v), g qualquer que o SymPy avalie numericamente."""
    try:
        if sympy.degree(sympy.Poly(f, v)) != 1:
            return None
    except sympy.PolynomialError:
        return None
    sols = sympy.solve(f, v)
    if len(sols) != 1:
        return None
    g = sols[0]
    try:
        fn = sympy.lambdify(h, g, "numpy")
        ys = fn(np.linspace(-3, 3, 7))
        if not np.all(np.isfinite(np.atleast_1d(ys) * 1.0)):
            pass  # descontinuidades são ok; o inkplot quebra a polilinha
    except Exception:  # noqa: BLE001 - função não avaliável numericamente
        return None
    return Analysis(
        "funcao_2d",
        f"grafico de {v.name} = {sympy.sstr(g)}",
        2,
        {"expr": sympy.sstr(g), "vars": (h.name, v.name)},
    )


# ── R³ ─────────────────────────────────────────────────────────────────────────────────

_KIND_3D_DESC = {
    "esfera": "esfera",
    "elipsoide": "elipsoide",
    "hiperboloide_1f": "hiperboloide de uma folha",
    "hiperboloide_2f": "hiperboloide de duas folhas",
    "cone": "cone",
    "paraboloide_eliptico": "paraboloide eliptico",
    "paraboloide_hiperbolico": "sela (paraboloide hiperbolico)",
}


def _analyze_3d(f: sympy.Expr) -> Analysis | None:
    try:
        poly = sympy.Poly(f, X, Y, Z)
        degree = poly.total_degree()
    except sympy.PolynomialError:
        return _function_3d(f)
    if degree == 1:
        return _plane(poly)
    if degree != 2:
        return _function_3d(f)

    def c(i: int, j: int, k: int) -> float:
        return float(poly.coeff_monomial(X**i * Y**j * Z**k) or 0)

    Q = np.array(
        [
            [c(2, 0, 0), c(1, 1, 0) / 2, c(1, 0, 1) / 2],
            [c(1, 1, 0) / 2, c(0, 2, 0), c(0, 1, 1) / 2],
            [c(1, 0, 1) / 2, c(0, 1, 1) / 2, c(0, 0, 2)],
        ]
    )
    b = np.array([c(1, 0, 0), c(0, 1, 0), c(0, 0, 1)])
    F = c(0, 0, 0)
    lam, R = np.linalg.eigh(Q)
    scale = max(float(np.max(np.abs(Q))), EPS)
    rank = int(np.sum(np.abs(lam) > 1e-9 * scale))

    if rank == 3:
        center = np.linalg.solve(2 * Q, -b)
        f0 = float(F + b @ center + center @ Q @ center)
        n_pos = int(np.sum(lam > 0))
        cen = f"({_fmt(center[0])}, {_fmt(center[1])}, {_fmt(center[2])})"
        if abs(f0) < EPS * scale:  # λ·t² = 0 → cone (assinatura mista) ou ponto
            if n_pos in (1, 2):
                return _quadric("cone", center, R, lam, f0, f"cone · vertice {cen}")
            return None
        ratios = -f0 / lam  # semi-eixo² ao longo de cada autovetor
        if np.all(ratios > 0):
            semi = np.sqrt(ratios)
            if np.ptp(lam) < 1e-6 * scale:
                return _quadric(
                    "esfera", center, R, lam, f0,
                    f"esfera · raio {_fmt(float(semi[0]))} · centro {cen}",
                )
            axes = " x ".join(_fmt(float(s)) for s in sorted(semi))
            return _quadric("elipsoide", center, R, lam, f0, f"elipsoide · semi-eixos {axes} · centro {cen}")
        if np.all(ratios < 0):
            return None  # vazio (ex.: x²+y²+z²=-1)
        n_pos_r = int(np.sum(ratios > 0))
        kind = "hiperboloide_1f" if n_pos_r == 2 else "hiperboloide_2f"
        return _quadric(kind, center, R, lam, f0, f"{_KIND_3D_DESC[kind]} · centro {cen}")

    if rank == 2:
        # paraboloide: termo linear na direção do autovalor nulo
        i0 = int(np.argmin(np.abs(lam)))
        axis = R[:, i0]
        e = float(b @ axis)
        if abs(e) < EPS * scale:
            return None  # cilindro (sem termo linear no eixo) — fora do escopo v1
        nz = [i for i in range(3) if i != i0]
        same_sign = lam[nz[0]] * lam[nz[1]] > 0
        kind = "paraboloide_eliptico" if same_sign else "paraboloide_hiperbolico"
        # vértice: minimiza nas direções não-nulas e resolve na direção do eixo
        u = np.array([-float(b @ R[:, i]) / (2 * lam[i]) if i != i0 else 0.0 for i in range(3)])
        vtx_partial = R @ u
        f_at = float(F + b @ vtx_partial + vtx_partial @ Q @ vtx_partial)
        t = -f_at / e
        vertex = vtx_partial + t * axis
        cen = f"({_fmt(vertex[0])}, {_fmt(vertex[1])}, {_fmt(vertex[2])})"
        return Analysis(
            kind,
            f"{_KIND_3D_DESC[kind]} · vertice {cen}",
            3,
            {
                "center": vertex.tolist(),
                "rot": R.tolist(),
                "lam": lam.tolist(),
                "axis_index": i0,
                "e": e,
            },
        )
    return None  # rank 1: cilindro parabólico / pares de planos — fora do escopo v1


def _quadric(kind: str, center, R, lam, f0: float, desc: str) -> Analysis:
    return Analysis(
        kind,
        desc,
        3,
        {"center": center.tolist(), "rot": R.tolist(), "lam": lam.tolist(), "f0": f0},
    )


def _plane(poly: sympy.Poly) -> Analysis:
    n = [float(poly.coeff_monomial(s) or 0) for s in (X, Y, Z)]
    d = float(poly.coeff_monomial(1) or 0)
    eq = " + ".join(f"{_fmt(c)}{s}" for c, s in zip(n, "xyz") if abs(c) > EPS)
    return Analysis(
        "plano",
        f"plano · {eq} = {_fmt(-d)}",
        3,
        {"normal": n, "d": d},
    )


def _function_3d(f: sympy.Expr) -> Analysis | None:
    """z = g(x, y) explícita, g avaliável numericamente."""
    try:
        if sympy.degree(sympy.Poly(f, Z)) != 1:
            return None
    except sympy.PolynomialError:
        return None
    sols = sympy.solve(f, Z)
    if len(sols) != 1:
        return None
    g = sols[0]
    try:
        fn = sympy.lambdify((X, Y), g, "numpy")
        fn(np.zeros(3), np.zeros(3))
    except Exception:  # noqa: BLE001
        return None
    return Analysis(
        "funcao_3d",
        f"superficie z = {sympy.sstr(g)}",
        3,
        {"expr": sympy.sstr(g)},
    )
