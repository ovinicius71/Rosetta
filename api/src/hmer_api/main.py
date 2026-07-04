"""App FastAPI: monta rotas e carrega o modelo no startup.

Fase 0/scaffold: as rotas existem e validam o contrato, mas devolvem 501 até haver
checkpoint treinado (Fase 3).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .evaluate import evaluate_latex
from .page import process_page
from .recognize import get_recognizer
from .schemas import (
    EvaluateRequest,
    EvaluateResponse,
    Ink,
    PageInk,
    PageProcessResponse,
    RecognizeResponse,
    SketchRecognizeResponse,
)
from .sketch import get_sketch_recognizer, recognize_sketch

app = FastAPI(title="HMER API", version="0.0.0")


@app.on_event("startup")
def _warmup() -> None:
    """Carrega os modelos já no boot: o 1º request não paga os ~20s de torch+checkpoint
    e não estoura o timeout do proxy do front."""
    get_recognizer()
    get_sketch_recognizer()

# CORS liberado p/ o dev do Next.js (localhost:3000). Restringir em produção.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(ink: Ink) -> RecognizeResponse:
    """Tinta → LaTeX. Delega ao Recognizer (hmer_ml.infer)."""
    recognizer = get_recognizer()
    latex = recognizer.recognize(ink.model_dump())
    return RecognizeResponse(latex=latex)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """LaTeX → resultado numérico/simbólico via SymPy (opcional)."""
    return evaluate_latex(req.latex)


@app.post("/page/process", response_model=PageProcessResponse)
def page_process(page: PageInk) -> PageProcessResponse:
    """Página do caderno → contas detectadas, reconhecidas e resolvidas (tinta pronta)."""
    return process_page(page)


@app.post("/sketch/recognize", response_model=SketchRecognizeResponse)
def sketch_recognize(page: PageInk) -> SketchRecognizeResponse:
    """Traços de um desenho → categoria (QuickDraw) + rótulo em tinta (Fase 4)."""
    return recognize_sketch(page)
