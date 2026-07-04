"""Testes de POST /sketch/recognize (classificador falso, sem checkpoint)."""

import pytest
from fastapi.testclient import TestClient
from hmer_ml.segment import RESULT_COLOR


def _circle(cx=100.0, cy=100.0, r=40.0, n=24):
    import math

    xs = [cx + r * math.cos(2 * math.pi * i / n) for i in range(n + 1)]
    ys = [cy + r * math.sin(2 * math.pi * i / n) for i in range(n + 1)]
    return {"x": xs, "y": ys, "width": 1.41}


class _FakeSketchRecognizer:
    def recognize(self, ink, topk=3):
        return [("circle", 0.91), ("moon", 0.05), ("sun", 0.02)]


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


def _with_fake(monkeypatch):
    import hmer_api.sketch as sketch_mod

    monkeypatch.setattr(sketch_mod, "get_sketch_recognizer", lambda: _FakeSketchRecognizer())
    # a rota importa o símbolo em main.py; o caminho de request usa recognize_sketch,
    # que resolve get_sketch_recognizer dentro de sketch.py — o patch acima basta.


def test_without_model_returns_501(client):
    r = client.post("/sketch/recognize", json={"strokes": [_circle()]})
    assert r.status_code == 501


def test_empty_returns_422(client, monkeypatch):
    _with_fake(monkeypatch)
    r = client.post("/sketch/recognize", json={"strokes": []})
    assert r.status_code == 422


def test_recognizes_and_labels_below_drawing(client, monkeypatch):
    _with_fake(monkeypatch)
    r = client.post("/sketch/recognize", json={"strokes": [_circle()]})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "circle"
    assert body["label_pt"] == "círculo"
    assert body["confidence"] == pytest.approx(0.91)
    assert [g["label"] for g in body["topk"]] == ["circle", "moon", "sun"]

    # rótulo em tinta: cor de resultado, abaixo do desenho (y > 140), centrado no cx
    strokes = body["strokes"]
    assert strokes
    assert all(s["color"] == RESULT_COLOR for s in strokes)
    ys = [y for s in strokes for y in s["y"]]
    assert min(ys) > 140  # borda inferior do círculo
    xs = [x for s in strokes for x in s["x"]]
    assert 60 < (min(xs) + max(xs)) / 2 < 140  # ~centrado em cx=100
