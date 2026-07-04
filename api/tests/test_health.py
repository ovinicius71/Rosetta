"""Testes da API. Rodar da raiz do repo: pytest api/tests (PYTHONPATH=api/src;ml/src)"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]
OVERFIT_CKPT = REPO / "checkpoints" / "overfit_crohme" / "last.ckpt"


@pytest.fixture()
def client(monkeypatch):
    """Client com o Recognizer stub (sem checkpoint) — contrato testável sem modelo."""
    monkeypatch.delenv("HMER_CKPT", raising=False)
    monkeypatch.setenv("SKETCH_CKPT", "")  # nunca carregar o classificador nos testes
    from hmer_api.main import app
    from hmer_api.recognize import get_recognizer
    from hmer_api.sketch import get_sketch_recognizer

    get_recognizer.cache_clear()
    get_sketch_recognizer.cache_clear()
    yield TestClient(app)
    get_recognizer.cache_clear()
    get_sketch_recognizer.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recognize_stub_returns_501_without_ckpt(client):
    payload = {"strokes": [{"points": [{"x": 1, "y": 2}]}]}
    r = client.post("/recognize", json=payload)
    assert r.status_code == 501  # contrato ok; modelo não configurado


def test_recognize_rejects_bad_payload(client):
    r = client.post("/recognize", json={"strokes": [{"points": []}]})
    assert r.status_code == 422  # min_length=1 do contrato


def test_evaluate_arithmetic(client):
    r = client.post("/evaluate", json={"latex": "1+1"})
    assert r.status_code == 200
    assert r.json()["result"] == "2"


def test_evaluate_fraction(client):
    r = client.post("/evaluate", json={"latex": r"\frac{3}{4}"})
    assert r.status_code == 200
    assert r.json()["result"].startswith("3/4")


def test_evaluate_solves_equation(client):
    r = client.post("/evaluate", json={"latex": "x + 1 = 3"})
    assert r.status_code == 200
    assert r.json()["result"] == "x = 2"


def test_evaluate_invalid_is_friendly(client):
    r = client.post("/evaluate", json={"latex": r"\frac{1}{"})
    assert r.status_code == 200
    assert r.json()["error"]  # erro amigável, não HTTP 500


@pytest.mark.skipif(not OVERFIT_CKPT.exists(), reason="requer checkpoint de overfit")
def test_recognize_with_real_model(monkeypatch):
    """Integração: carrega o modelo real e reconhece a tinta de uma amostra do CROHME."""
    monkeypatch.setenv("HMER_CKPT", str(OVERFIT_CKPT))
    monkeypatch.setenv("HMER_CONFIG", str(REPO / "ml" / "configs" / "overfit_crohme.yaml"))
    monkeypatch.chdir(REPO)  # vocab_path do config é relativo à raiz
    from hmer_api.main import app
    from hmer_api.recognize import get_recognizer

    get_recognizer.cache_clear()
    try:
        from hmer_ml.data.inkml import iter_inkml

        _, ink = next(iter_inkml(REPO / "data" / "crohme" / "train"))
        payload = ink.to_dict()
        payload.pop("label", None)  # inferência não recebe label
        r = TestClient(app).post("/recognize", json=payload)
        assert r.status_code == 200
        assert isinstance(r.json()["latex"], str) and r.json()["latex"]
    finally:
        get_recognizer.cache_clear()
