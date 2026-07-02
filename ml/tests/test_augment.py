"""Testes de augmentation de tinta. Python puro (sem torch)."""

import math
import pickle
import random

from hmer_ml.data.augment import (
    Augmenter,
    Jitter,
    PointDropout,
    Rotate,
    Scale,
    build_augmenter,
)
from hmer_ml.data.ink import Ink, Point, Stroke


def _ink():
    return Ink(
        strokes=[
            Stroke([Point(0, 0, 0), Point(10, 0, 16), Point(10, 10, 33)]),
            Stroke([Point(20, 20), Point(30, 25)]),
        ],
        label="x + 1",
    )


def test_jitter_preserves_structure():
    rng = random.Random(0)
    out = Jitter(sigma=0.01)(_ink(), rng)
    assert len(out.strokes) == 2
    assert [len(s.points) for s in out.strokes] == [3, 2]
    assert out.label == "x + 1"  # metadados preservados
    assert out.strokes[0].points[1].t == 16  # timestamps preservados
    # mexeu nas coordenadas, mas pouco (sigma relativo à bbox de 30 → ~0.3)
    assert out.strokes[0].points[0].x != 0 or out.strokes[0].points[0].y != 0
    assert abs(out.strokes[0].points[0].x) < 3


def test_rotate_preserves_distances_to_center():
    rng = random.Random(1)
    ink = _ink()
    out = Rotate(max_deg=8.0)(ink, rng)
    # rotação é isometria: distância entre dois pontos se preserva
    a, b = ink.strokes[0].points[0], ink.strokes[0].points[2]
    a2, b2 = out.strokes[0].points[0], out.strokes[0].points[2]
    d1 = math.hypot(a.x - b.x, a.y - b.y)
    d2 = math.hypot(a2.x - b2.x, a2.y - b2.y)
    assert abs(d1 - d2) < 1e-6


def test_scale_is_anisotropic_but_bounded():
    rng = random.Random(2)
    out = Scale(0.5, 0.6)(_ink(), rng)  # range estreito p/ testar limites
    xs = [p.x for s in out.strokes for p in s.points]
    w = max(xs) - min(xs)
    assert 0.5 * 30 <= w <= 0.6 * 30 + 1e-9  # largura original 30


def test_point_dropout_keeps_endpoints():
    rng = random.Random(3)
    ink = Ink(strokes=[Stroke([Point(i, i) for i in range(50)])])
    out = PointDropout(p=0.5)(ink, rng)
    pts = out.strokes[0].points
    assert pts[0].x == 0 and pts[-1].x == 49  # extremos preservados
    assert 2 <= len(pts) < 50  # removeu algo


def test_augmenter_is_picklable():
    """Windows DataLoader (spawn) serializa o dataset — o augmenter precisa picklar."""
    aug = build_augmenter({"jitter_sigma": 0.01, "seed": 42})
    restored = pickle.loads(pickle.dumps(aug))
    out = restored(_ink())
    assert isinstance(out, Ink)


def test_build_augmenter_disables_with_zero():
    aug = build_augmenter(
        {"jitter_sigma": 0, "scale_range": None, "rotate_max_deg": 0, "point_dropout": 0}
    )
    assert isinstance(aug, Augmenter)
    assert aug.transforms == []  # tudo desligado
    ink = _ink()
    out = aug(ink)
    assert out.strokes[0].points[0].x == ink.strokes[0].points[0].x
