"""Smoke tests do scaffold. Não exigem torch nem datasets — validam contratos leves.

Rodar: uv run pytest ml/tests
"""

import json
from pathlib import Path

from hmer_ml.data.ink import (
    NUM_FEATURES,
    Ink,
    Point,
    Stroke,
    ink_to_features,
    normalize,
    resample,
)
from hmer_ml.data.inkml import parse_inkml
from hmer_ml.tokenizer import LatexTokenizer

REPO = Path(__file__).resolve().parents[2]

# InkML no formato CROHME: truth da expressão + <traceGroup> com truths por símbolo.
_CROHME_INKML = """<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="UI">dummy</annotation>
  <annotation type="truth">$\\frac { 1 } { 2 }$</annotation>
  <trace id="0">0 0, 1 1</trace>
  <trace id="1">5 5, 6 4</trace>
  <traceGroup xml:id="tg">
    <annotation type="truth">Segmentation</annotation>
    <traceGroup xml:id="tg0">
      <annotation type="truth">\\frac</annotation>
      <traceView traceDataRef="0"/>
    </traceGroup>
  </traceGroup>
</ink>"""


def test_ink_from_dict_matches_shared_example():
    """O exemplo canônico do esquema compartilhado carrega no espelho Python."""
    example = json.loads((REPO / "schemas" / "ink_example.json").read_text(encoding="utf-8"))
    ink = Ink.from_dict(example)
    assert len(ink.strokes) == 2
    assert ink.strokes[0].points[0].x == 120.0
    assert ink.label == "x^2"


def test_feature_layout_is_stable():
    """in_features nas configs (6) precisa casar com FEATURE_NAMES."""
    assert NUM_FEATURES == 6


def test_ink_to_features_shape_and_boundaries():
    """Features têm shape [T, 6]; dx/dy resetam por traço; eos_stroke marca fim de traço."""
    ink = Ink(
        strokes=[
            Stroke([Point(0, 0), Point(1, 1)]),
            Stroke([Point(5, 5)]),
        ]
    )
    feats = ink_to_features(ink)
    assert len(feats) == 3
    assert all(len(row) == NUM_FEATURES for row in feats)
    # 1º ponto de cada traço tem dx=dy=0
    assert feats[0][2] == 0.0 and feats[0][3] == 0.0
    assert feats[2][2] == 0.0 and feats[2][3] == 0.0
    # segundo ponto do 1º traço: dx=dy=1
    assert feats[1][2] == 1.0 and feats[1][3] == 1.0
    # eos_stroke: 1 no fim de cada traço
    assert feats[1][5] == 1.0 and feats[2][5] == 1.0
    assert feats[0][5] == 0.0


def test_normalize_centers_and_scales():
    """Após normalize, o maior lado da bbox ocupa ~[-1, 1] e fica centrado em 0."""
    ink = Ink(strokes=[Stroke([Point(10, 10), Point(30, 20)])])
    norm = normalize(ink)
    xs = [p.x for s in norm.strokes for p in s.points]
    ys = [p.y for s in norm.strokes for p in s.points]
    assert min(xs) == -1.0 and max(xs) == 1.0  # x é o maior lado (20)
    assert abs((min(ys) + max(ys)) / 2) < 1e-9  # centrado


def test_resample_uniform_spacing():
    """Passo ~constante ao longo do arco, atravessando segmentos de tamanhos variados."""
    ink = Ink(strokes=[Stroke([Point(0, 0), Point(0.003, 0), Point(1, 0)])])
    pts = resample(ink, 0.25).strokes[0].points
    xs = [p.x for p in pts]
    assert xs == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_resample_is_density_invariant():
    """Interpolar pontos colineares no meio (caneta mais densa) NÃO muda a saída.

    Regressão do bug em que `carry` não acumulava entre segmentos curtos: tinta mais
    densa que o passo emitia MENOS pontos — o oposto do contrato do resample.
    """
    sparse = Ink(strokes=[Stroke([Point(0, 0), Point(1, 0), Point(1, 1)])])
    dense = Ink(
        strokes=[
            Stroke(
                [Point(i / 100, 0) for i in range(101)]
                + [Point(1, i / 100) for i in range(1, 101)]
            )
        ]
    )
    r_sparse = resample(sparse, 0.04).strokes[0].points
    r_dense = resample(dense, 0.04).strokes[0].points
    assert len(r_sparse) == len(r_dense)
    assert all(
        abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9
        for a, b in zip(r_sparse, r_dense)
    )
    # e o passo é de fato ~constante
    import math

    gaps = [
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(r_dense, r_dense[1:])
    ]
    assert all(abs(g - 0.04) < 1e-6 for g in gaps[:-1])  # último vai até o ponto final


def test_tokenizer_roundtrip():
    """tokenize→encode→decode reproduz os tokens (com espaço como separador)."""
    tok = LatexTokenizer().build_vocab([r"\frac { 1 } { 2 }", r"x ^ 2 + \alpha"])
    ids = tok.encode(r"\frac { 1 } { 2 }", add_special=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == r"\frac { 1 } { 2 }"


def test_tokenizer_command_is_single_token():
    r"""\frac e \alpha são 1 token cada; ^ e _ são tokens próprios."""
    assert LatexTokenizer.tokenize(r"\frac{x}^2") == [r"\frac", "{", "x", "}", "^", "2"]


def test_tokenizer_unknown_maps_to_unk():
    tok = LatexTokenizer().build_vocab(["x + 1"])
    ids = tok.encode(r"\zeta", add_special=False)
    assert ids == [tok.unk_id]


def test_inkml_expression_label_ignores_symbol_truths(tmp_path):
    """No CROHME, o label é a expressão inteira — não o truth de um símbolo do traceGroup."""
    f = tmp_path / "s.inkml"
    f.write_text(_CROHME_INKML, encoding="utf-8")
    ink = parse_inkml(f)
    assert ink.label == r"\frac { 1 } { 2 }"  # sem $, expressão inteira
    assert len(ink.strokes) == 2  # <traceGroup> não vira stroke


# TODO(Fase 1): test de shapes do collate_fn e de um forward do InkModel (requer torch).
# TODO(ADR 0006): test garantindo que o encoder não referencia símbolos de LaTeX.
