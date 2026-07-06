"""Gráficos como tinta: Analysis → polilinhas (eixos + curva/superfície) prontas p/ a página.

R²: eixos cartesianos com setas/ticks + curva amostrada parametricamente.
R³: projeção isométrica fixa, 3 eixos rotulados + wireframe esparso (estética de esboço,
sem remoção de linha oculta — combina com o caderno).

Tudo em "world coords" primeiro; um único mapeamento no fim leva à página (y para baixo),
preservando proporção (círculo continua redondo). Só numpy + inkfont (rótulos Hershey).
"""

from __future__ import annotations

import math

import numpy as np
import sympy

from .analyze import Analysis
from .inkfont import text_to_strokes

Polyline = list[tuple[float, float]]

# calibração
SAMPLES = 64  # pontos por curva fechada
PAD = 0.18  # folga da janela em torno da curva
ARROW = 0.05  # tamanho da seta relativo à extensão da janela
FN_WINDOW = 4.0  # janela de y=f(x) / z=f(x,y)
AZ, EL = math.radians(-55.0), math.radians(22.0)  # câmera isométrica (R³)


def analysis_to_strokes(analysis: Analysis, *, x: float, y: float, size: float) -> list[Polyline]:
    """Desenha o gráfico num quadro de largura `size` com canto superior esquerdo (x, y)."""
    if analysis.dim == 2:
        curves, labels = _plot_2d(analysis)
    else:
        curves, labels = _plot_3d(analysis)
    return _to_page(curves, labels, x=x, y=y, size=size)


# ── R² ─────────────────────────────────────────────────────────────────────────────────


def _plot_2d(a: Analysis) -> tuple[list[np.ndarray], list[tuple[float, float, str]]]:
    """Curvas em world coords (y para CIMA) + rótulos (wx, wy, texto)."""
    p = a.params
    curves: list[np.ndarray] = []

    if a.kind in ("circunferencia", "elipse", "hiperbole"):
        c = np.array(p["center"])
        a1, a2 = p["axes"]
        ang = p.get("angle", 0.0)
        e1 = np.array([math.cos(ang), math.sin(ang)])
        e2 = np.array([-math.sin(ang), math.cos(ang)])
        if a.kind == "hiperbole":
            t = np.linspace(-1.6, 1.6, SAMPLES // 2)
            for sign in (+1, -1):
                pts = c + sign * a1 * np.cosh(t)[:, None] * e1 + a2 * np.sinh(t)[:, None] * e2
                curves.append(pts)
        else:
            th = np.linspace(0, 2 * math.pi, SAMPLES + 1)
            curves.append(c + a1 * np.cos(th)[:, None] * e1 + a2 * np.sin(th)[:, None] * e2)
    elif a.kind == "parabola":
        v = np.array(p["vertex"])
        u_dir, w_dir = np.array(p["u_dir"]), np.array(p["w_dir"])
        k = p["k"]
        span = max(1.0, min(6.0, math.sqrt(4.0 / max(abs(k), 1e-6))))
        u = np.linspace(-span, span, SAMPLES)
        curves.append(v + u[:, None] * u_dir + (k * u**2)[:, None] * w_dir)
    elif a.kind == "reta":
        aa, bb, cc = p["a"], p["b"], p["c"]
        W = 3.0
        if abs(bb) > 1e-9:
            xs = np.array([-W, W])
            curves.append(np.stack([xs, (-aa * xs - cc) / bb], axis=1))
        else:
            x0 = -cc / aa
            curves.append(np.array([[x0, -W], [x0, W]]))
    elif a.kind == "funcao_2d":
        h, v = sympy.symbols(" ".join(p["vars"]))
        fn = sympy.lambdify(h, sympy.sympify(p["expr"]), "numpy")
        xs = np.linspace(-FN_WINDOW, FN_WINDOW, 160)
        with np.errstate(all="ignore"):
            ys = np.asarray(fn(xs), dtype=float)
            if ys.ndim == 0:
                ys = np.full_like(xs, float(ys))
        ys[~np.isfinite(ys)] = np.nan
        ys[np.abs(ys) > FN_WINDOW * 2] = np.nan
        curves.extend(_split_nan(np.stack([xs, ys], axis=1)))

    window = _window_2d(curves)
    axes, labels = _axes_2d(window, a.params.get("vars", ("x", "y")))
    return axes + curves, labels


def _split_nan(pts: np.ndarray) -> list[np.ndarray]:
    """Quebra a polilinha em segmentos finitos (descontinuidades viram lacunas)."""
    out, cur = [], []
    for row in pts:
        if np.all(np.isfinite(row)):
            cur.append(row)
        elif len(cur) >= 2:
            out.append(np.array(cur))
            cur = []
        else:
            cur = []
    if len(cur) >= 2:
        out.append(np.array(cur))
    return out


def _window_2d(curves: list[np.ndarray]) -> tuple[float, float, float, float]:
    """Janela = bbox das curvas ∪ origem (os eixos sempre aparecem), com folga."""
    pts = np.concatenate([c for c in curves if len(c)] + [np.zeros((1, 2))])
    x0, y0 = np.nanmin(pts, axis=0)
    x1, y1 = np.nanmax(pts, axis=0)
    span = max(x1 - x0, y1 - y0, 1e-6)
    pad = PAD * span
    return float(x0 - pad), float(x1 + pad), float(y0 - pad), float(y1 + pad)


def _nice_step(span: float) -> float:
    raw = span / 4.0
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _axes_2d(window, var_names) -> tuple[list[np.ndarray], list[tuple[float, float, str]]]:
    x0, x1, y0, y1 = window
    ah = ARROW * max(x1 - x0, y1 - y0)
    curves = [
        np.array([[x0, 0.0], [x1, 0.0]]),  # eixo horizontal
        np.array([[x1 - ah, ah / 2], [x1, 0.0], [x1 - ah, -ah / 2]]),  # seta →
        np.array([[0.0, y0], [0.0, y1]]),  # eixo vertical
        np.array([[-ah / 2, y1 - ah], [0.0, y1], [ah / 2, y1 - ah]]),  # seta ↑
    ]
    labels = [(x1, -3.5 * ah, var_names[0]), (2.0 * ah, y1, var_names[1])]
    # ticks + um número de escala em cada eixo positivo
    step = _nice_step(max(x1 - x0, y1 - y0))
    tick = ah * 0.45
    t = step
    while t < x1 - ah:
        curves.append(np.array([[t, -tick], [t, tick]]))
        curves.append(np.array([[-t, -tick], [-t, tick]])) if -t > x0 else None
        t += step
    t = step
    while t < y1 - ah:
        curves.append(np.array([[-tick, t], [tick, t]]))
        curves.append(np.array([[-tick, -t], [tick, -t]])) if -t > y0 else None
        t += step
    num = f"{step:g}"
    labels.append((step, -3.5 * ah, num))
    return curves, labels


# ── R³ ─────────────────────────────────────────────────────────────────────────────────


def _project(P: np.ndarray) -> np.ndarray:
    """[N,3] world → [N,2] tela isométrica (y da tela para CIMA; _to_page inverte)."""
    ca, sa, ce, se = math.cos(AZ), math.sin(AZ), math.cos(EL), math.sin(EL)
    xr = P[:, 0] * ca - P[:, 1] * sa
    yr = P[:, 0] * sa + P[:, 1] * ca
    sx = xr
    sy = P[:, 2] * ce - yr * se
    return np.stack([sx, sy], axis=1)


def _plot_3d(a: Analysis) -> tuple[list[np.ndarray], list[tuple[float, float, str]]]:
    p = a.params
    curves3d: list[np.ndarray] = []

    if a.kind in ("esfera", "elipsoide", "hiperboloide_1f", "hiperboloide_2f", "cone"):
        c = np.array(p["center"])
        R = np.array(p["rot"])
        lam = np.array(p["lam"])
        f0 = p["f0"]
        curves3d = _quadric_wireframe(a.kind, c, R, lam, f0)
    elif a.kind in ("paraboloide_eliptico", "paraboloide_hiperbolico"):
        curves3d = _paraboloid_wireframe(p)
    elif a.kind == "plano":
        curves3d = _plane_wireframe(p)
    elif a.kind == "funcao_3d":
        curves3d = _surface_wireframe(p["expr"])

    extent = max(float(np.max(np.abs(np.concatenate(curves3d)))), 1e-6)
    L = extent * 1.25
    axes3d = [
        np.array([[0.0, 0, 0], [L, 0, 0]]),
        np.array([[0.0, 0, 0], [0, L, 0]]),
        np.array([[0.0, 0, 0], [0, 0, L]]),
    ]
    ends2d = _project(np.array([[L, 0, 0], [0, L, 0], [0, 0, L]]) * 1.12)
    labels = [(float(e[0]), float(e[1]), n) for e, n in zip(ends2d, ("x", "y", "z"))]

    curves = [_project(c3) for c3 in axes3d + curves3d]
    return curves, labels


def _ring(radii: tuple[float, float], n: int = 48) -> np.ndarray:
    th = np.linspace(0, 2 * math.pi, n + 1)
    return np.stack([radii[0] * np.cos(th), radii[1] * np.sin(th)], axis=1)


def _quadric_wireframe(kind, c, R, lam, f0) -> list[np.ndarray]:
    """Wireframe no referencial dos autovetores → world (R @ p + c)."""

    def to_world(pts_eig: np.ndarray) -> np.ndarray:
        return pts_eig @ R.T + c

    out: list[np.ndarray] = []
    if kind in ("esfera", "elipsoide"):
        semi = np.sqrt(-f0 / lam)
        th = np.linspace(0, 2 * math.pi, 49)
        for phi in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):  # meridianos
            u = np.cos(th)[:, None] * np.array([math.cos(phi), math.sin(phi), 0.0])
            w = np.sin(th)[:, None] * np.array([0.0, 0.0, 1.0])
            out.append(to_world((u + w) * semi))
        for zf in (-0.55, 0.0, 0.55):  # paralelos
            r = math.sqrt(1 - zf**2)
            ring = _ring((r * semi[0], r * semi[1]))
            out.append(to_world(np.column_stack([ring, np.full(len(ring), zf * semi[2])])))
        return out

    if kind == "cone":
        # λ1u²+λ2v²+λ3w²=0; eixo = direção do autovalor de sinal minoritário
        i_ax = int(np.argmin(lam * np.sign(np.sum(np.sign(lam)))))
        others = [i for i in range(3) if i != i_ax]
        for wv in (-1.0, 1.0):
            radii = tuple(math.sqrt(-lam[i_ax] / lam[i]) * abs(wv) for i in others)
            ring = _ring(radii)
            pts = np.zeros((len(ring), 3))
            pts[:, others] = ring
            pts[:, i_ax] = wv
            out.append(to_world(pts))
        ring = _ring((math.sqrt(-lam[i_ax] / lam[others[0]]), math.sqrt(-lam[i_ax] / lam[others[1]])), n=4)
        for q in ring[:4]:  # 4 geratrizes passando pelo vértice
            g = np.zeros((2, 3))
            g[0, others] = -q
            g[0, i_ax] = -1.0
            g[1, others] = q
            g[1, i_ax] = 1.0
            out.append(to_world(g))
        return out

    # hiperboloides: assinatura de -f0/λ decide o eixo
    ratios = -f0 / lam
    if kind == "hiperboloide_1f":
        i_ax = int(np.argmin(ratios))  # razão negativa = eixo
        others = [i for i in range(3) if i != i_ax]
        aa = [math.sqrt(ratios[i]) for i in others]
        cc = math.sqrt(-ratios[i_ax])
        for t in (-1.0, 0.0, 1.0):
            r = math.sqrt(1 + t**2)
            ring = _ring((aa[0] * r, aa[1] * r))
            pts = np.zeros((len(ring), 3))
            pts[:, others] = ring
            pts[:, i_ax] = cc * t
            out.append(to_world(pts))
        v = np.linspace(-1.2, 1.2, 25)
        for phi in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
            pts = np.zeros((len(v), 3))
            pts[:, others[0]] = aa[0] * np.cosh(v) * math.cos(phi)
            pts[:, others[1]] = aa[1] * np.cosh(v) * math.sin(phi)
            pts[:, i_ax] = cc * np.sinh(v)
            out.append(to_world(pts))
        return out

    # duas folhas: eixo = razão positiva
    i_ax = int(np.argmax(ratios))
    others = [i for i in range(3) if i != i_ax]
    aa = [math.sqrt(-ratios[i]) for i in others]
    cc = math.sqrt(ratios[i_ax])
    v = np.linspace(0, 1.2, 13)
    for sign in (-1.0, 1.0):
        for t in (0.6, 1.1):
            ring = _ring((aa[0] * math.sinh(t), aa[1] * math.sinh(t)))
            pts = np.zeros((len(ring), 3))
            pts[:, others] = ring
            pts[:, i_ax] = sign * cc * math.cosh(t)
            out.append(to_world(pts))
        for phi in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
            pts = np.zeros((len(v), 3))
            pts[:, others[0]] = aa[0] * np.sinh(v) * math.cos(phi)
            pts[:, others[1]] = aa[1] * np.sinh(v) * math.sin(phi)
            pts[:, i_ax] = sign * cc * np.cosh(v)
            out.append(to_world(pts))
    return out


def _paraboloid_wireframe(p: dict) -> list[np.ndarray]:
    c = np.array(p["center"])
    R = np.array(p["rot"])
    lam = np.array(p["lam"])
    i0 = p["axis_index"]
    e = p["e"]
    others = [i for i in range(3) if i != i0]
    l1, l2 = lam[others[0]], lam[others[1]]

    def to_world(u, v):
        w = -(l1 * u**2 + l2 * v**2) / e
        pts = np.zeros((len(np.atleast_1d(u)), 3))
        pts[:, others[0]] = u
        pts[:, others[1]] = v
        pts[:, i0] = w
        return pts @ R.T + c

    out = []
    span = 1.6
    grid = np.linspace(-span, span, 7)
    t = np.linspace(-span, span, 25)
    for g in grid:  # curvas u=const e v=const (funciona p/ elíptico e sela)
        out.append(to_world(np.full_like(t, g), t))
        out.append(to_world(t, np.full_like(t, g)))
    return out


def _plane_wireframe(p: dict) -> list[np.ndarray]:
    n = np.array(p["normal"], dtype=float)
    d = p["d"]
    n_hat = n / np.linalg.norm(n)
    p0 = -d * n_hat / np.linalg.norm(n)  # ponto do plano mais próximo da origem
    ref = np.array([0.0, 0, 1]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(n_hat, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    span = max(2.0, float(np.linalg.norm(p0)) * 1.2)
    grid = np.linspace(-span, span, 5)
    out = []
    for g in grid:
        out.append(np.stack([p0 + g * e1 + t * e2 for t in (-span, span)]))
        out.append(np.stack([p0 + t * e1 + g * e2 for t in (-span, span)]))
    return out


def _surface_wireframe(expr_str: str) -> list[np.ndarray]:
    x, y = sympy.symbols("x y")
    fn = sympy.lambdify((x, y), sympy.sympify(expr_str), "numpy")
    span = 2.0
    grid = np.linspace(-span, span, 5)
    t = np.linspace(-span, span, 25)
    out = []
    with np.errstate(all="ignore"):
        for g in grid:
            for u, v in ((np.full_like(t, g), t), (t, np.full_like(t, g))):
                z = np.asarray(fn(u, v), dtype=float)
                if z.ndim == 0:
                    z = np.full_like(t, float(z))
                z[np.abs(z) > span * 2.5] = np.nan
                pts = np.stack([u, v, z], axis=1)
                out.extend(
                    seg for seg in _split_nan_3d(pts) if len(seg) >= 2
                )
    return out


def _split_nan_3d(pts: np.ndarray) -> list[np.ndarray]:
    out, cur = [], []
    for row in pts:
        if np.all(np.isfinite(row)):
            cur.append(row)
        else:
            if len(cur) >= 2:
                out.append(np.array(cur))
            cur = []
    if len(cur) >= 2:
        out.append(np.array(cur))
    return out


# ── mapeamento world → página ──────────────────────────────────────────────────────────


def _to_page(
    curves: list[np.ndarray],
    labels: list[tuple[float, float, str]],
    *,
    x: float,
    y: float,
    size: float,
) -> list[Polyline]:
    curves = [c for c in curves if len(c) >= 2]
    if not curves:
        return []
    allp = np.concatenate(curves)
    x0, y0 = np.min(allp, axis=0)
    x1, y1 = np.max(allp, axis=0)
    s = size / max(x1 - x0, y1 - y0, 1e-9)  # escala uniforme (círculo fica redondo)

    def map_xy(wx: float, wy: float) -> tuple[float, float]:
        return x + (wx - x0) * s, y + (y1 - wy) * s  # world y↑ → página y↓

    out: list[Polyline] = [[map_xy(px, py) for px, py in c] for c in curves]
    label_h = max(10.0, min(0.06 * size, 16.0))
    for wx, wy, text in labels:
        lx, ly = map_xy(wx, wy)
        out.extend(
            [(px, py) for px, py in poly]
            for poly in text_to_strokes(text, x=lx, y_mid=ly, height=label_h)
        )
    return out
