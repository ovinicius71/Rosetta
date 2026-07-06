"""Testes de POST /page/process (integração Xournal++), com recognizer falso."""

import pytest
from fastapi.testclient import TestClient
from hmer_ml.segment import RESULT_COLOR


def _line(x0, y0, x1, y1, n=8):
    return {
        "x": [x0 + (x1 - x0) * i / (n - 1) for i in range(n)],
        "y": [y0 + (y1 - y0) * i / (n - 1) for i in range(n)],
        "width": 1.41,
    }


def _zigzag(cx, cy, w=10.0, h=14.0):
    xs = [cx - w / 2, cx + w / 2, cx - w / 2, cx + w / 2, cx - w / 2]
    ys = [cy - h / 2, cy - h / 4, cy, cy + h / 4, cy + h / 2]
    return {"x": xs, "y": ys, "width": 1.41}


def _page_2_mais_3():
    """'2 + 3 =' na linha y≈50 (mesma geometria dos testes de hmer_ml.segment)."""
    return {
        "strokes": [
            _zigzag(20, 50),  # "2"
            _line(35, 50, 45, 50),  # barra horizontal do "+"
            _line(40, 44, 40, 56),  # barra vertical do "+"
            _zigzag(60, 50),  # "3"
            _line(75, 47, 87, 47),  # "=" (barra de cima)
            _line(75, 52, 87, 52),  # "=" (barra de baixo)
        ]
    }


class _FakeRecognizer:
    def __init__(self, latex):
        self.latex = latex
        self.calls = []

    def recognize(self, ink: dict) -> str:
        self.calls.append(ink)
        return self.latex


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("HMER_CKPT", raising=False)
    monkeypatch.setenv("SKETCH_CKPT", "")
    from hmer_api.main import app
    from hmer_api.recognize import get_recognizer
    from hmer_api.sketch import get_sketch_recognizer

    get_recognizer.cache_clear()
    get_sketch_recognizer.cache_clear()
    yield TestClient(app)
    get_recognizer.cache_clear()
    get_sketch_recognizer.cache_clear()


def _with_fake(monkeypatch, latex):
    import hmer_api.page as page_mod

    fake = _FakeRecognizer(latex)
    monkeypatch.setattr(page_mod, "get_recognizer", lambda: fake)
    return fake


def test_empty_page_needs_no_model(client):
    r = client.post("/page/process", json={"strokes": []})
    assert r.status_code == 200
    assert r.json() == {"expressions": []}


def test_page_with_conta_but_no_model_returns_501(client):
    r = client.post("/page/process", json=_page_2_mais_3())
    assert r.status_code == 501


def test_recognizes_and_draws_result(client, monkeypatch):
    fake = _with_fake(monkeypatch, "2 + 3")
    r = client.post("/page/process", json=_page_2_mais_3())
    assert r.status_code == 200
    exprs = r.json()["expressions"]
    assert len(exprs) == 1
    assert exprs[0]["latex"] == "2 + 3"
    assert exprs[0]["result"] == "5"

    # só o lado esquerdo (4 traços) vai ao modelo — os 2 traços do '=' ficam de fora
    assert len(fake.calls) == 1
    sent = fake.calls[0]["strokes"]
    assert len(sent) == 4

    # a tinta vai CRUA (coords da página): normalize+resample são do Recognizer,
    # que aplica o mesmo pré-processamento do treino (via config)
    pts = [p for s in sent for p in s["points"]]
    assert min(p["x"] for p in pts) == pytest.approx(15.0)  # borda esquerda do "2"
    assert max(p["x"] for p in pts) == pytest.approx(65.0)  # borda direita do "3"

    # tinta do resultado: cor de marcação, à direita do '=', na mesma linha
    strokes = exprs[0]["strokes"]
    assert strokes
    assert all(s["color"] == RESULT_COLOR for s in strokes)
    assert min(x for s in strokes for x in s["x"]) > 87  # borda direita do '='
    ys = [y for s in strokes for y in s["y"]]
    assert 40 < (min(ys) + max(ys)) / 2 < 60  # centrado na linha de escrita
    assert all(s["width"] == 1.41 for s in strokes)  # casa com a caneta


def test_lone_equals_is_not_a_conta(client, monkeypatch):
    fake = _with_fake(monkeypatch, "2 + 3")
    page = {"strokes": [_line(75, 47, 87, 47), _line(75, 52, 87, 52)]}  # só um '='
    r = client.post("/page/process", json=page)
    assert r.status_code == 200
    assert r.json()["expressions"] == []
    assert fake.calls == []  # modelo nem foi chamado


def test_second_pass_is_idempotent(client, monkeypatch):
    _with_fake(monkeypatch, "2 + 3")
    page = _page_2_mais_3()
    first = client.post("/page/process", json=page).json()["expressions"]
    page["strokes"] += [
        {"x": s["x"], "y": s["y"], "color": s["color"], "width": s["width"]}
        for s in first[0]["strokes"]
    ]
    r = client.post("/page/process", json=page)
    assert r.status_code == 200
    assert r.json()["expressions"] == []


def test_unevaluable_latex_reports_error_without_ink(client, monkeypatch):
    _with_fake(monkeypatch, "\\frac{1}{")  # LaTeX quebrado: reconheceu mas não avalia
    r = client.post("/page/process", json=_page_2_mais_3())
    assert r.status_code == 200
    exprs = r.json()["expressions"]
    assert len(exprs) == 1
    assert exprs[0]["result"] is None
    assert exprs[0]["error"]
    assert exprs[0]["strokes"] == []


def _page_equacao():
    """'<x> <y> = <4>' na linha y≈50 — equação com lado direito."""
    return {
        "strokes": [
            _zigzag(20, 50),  # "x²+y²" (geometria genérica)
            _zigzag(45, 50),
            _line(60, 47, 72, 47),  # "="
            _line(60, 52, 72, 52),
            _zigzag(95, 50),  # "4"
        ]
    }


def test_equation_is_classified_and_plotted_below(client, monkeypatch):
    fake = _with_fake(monkeypatch, "x ^ 2 + y ^ 2 = 4")
    r = client.post("/page/process", json=_page_equacao())
    assert r.status_code == 200
    exprs = r.json()["expressions"]
    assert len(exprs) == 1
    e = exprs[0]
    assert e["kind"] == "circunferencia"
    assert "raio 2" in e["description"]
    assert e["result"] is None

    # a equação COMPLETA (5 traços, com o '=') foi ao modelo
    assert len(fake.calls[0]["strokes"]) == 5

    # gráfico desenhado ABAIXO da equação (equação termina em y≈57)
    strokes = e["strokes"]
    assert len(strokes) > 5  # eixos + curva + descrição
    assert all(s["color"] == RESULT_COLOR for s in strokes)
    assert min(y for s in strokes for y in s["y"]) > 57


def test_equation_second_pass_is_idempotent(client, monkeypatch):
    _with_fake(monkeypatch, "x ^ 2 + y ^ 2 = 4")
    page = _page_equacao()
    first = client.post("/page/process", json=page).json()["expressions"]
    page["strokes"] += [
        {"x": s["x"], "y": s["y"], "color": s["color"], "width": s["width"]}
        for s in first[0]["strokes"]
    ]
    r = client.post("/page/process", json=page)
    assert r.status_code == 200
    assert r.json()["expressions"] == []


class _FakeTopkRecognizer(_FakeRecognizer):
    def __init__(self, latexes):
        super().__init__(latexes[0])
        self.latexes = latexes

    def recognize_topk(self, ink: dict) -> list[str]:
        self.calls.append(ink)
        return self.latexes


def test_equation_uses_first_classifiable_beam_candidate(client, monkeypatch):
    """A melhor hipótese não classifica (x²+x²=8, 1 variável) → a 2ª (circunferência)
    vence. É o que salva leituras com 1 símbolo trocado."""
    import hmer_api.page as page_mod

    fake = _FakeTopkRecognizer(["X ^ 2 + X ^ 2 = 8", "x ^ 2 + y ^ 2 = 4"])
    monkeypatch.setattr(page_mod, "get_recognizer", lambda: fake)
    r = client.post("/page/process", json=_page_equacao())
    assert r.status_code == 200
    e = r.json()["expressions"][0]
    assert e["kind"] == "circunferencia"
    assert e["latex"] == "x ^ 2 + y ^ 2 = 4"


def test_one_var_equation_solves_below(client, monkeypatch):
    _with_fake(monkeypatch, "2 x + 4 = 10")
    r = client.post("/page/process", json=_page_equacao())
    assert r.status_code == 200
    e = r.json()["expressions"][0]
    assert e["kind"] is None
    assert e["result"] == "x = 3"
    assert e["strokes"]
    assert min(y for s in e["strokes"] for y in s["y"]) > 57  # abaixo, não à direita


def test_mismatched_xy_rejected(client):
    r = client.post("/page/process", json={"strokes": [{"x": [1, 2], "y": [1]}]})
    assert r.status_code == 422
