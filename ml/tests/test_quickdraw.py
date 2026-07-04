"""Testes do pipeline QuickDraw (parse, split, features)."""

import json

import pytest

torch = pytest.importorskip("torch")

from hmer_ml.data.quickdraw import (  # noqa: E402
    QuickDrawDataset,
    SketchCollate,
    parse_quickdraw_line,
    prepare_sketch,
)


def _line(word: str, xs, ys, recognized: bool = True) -> str:
    return json.dumps(
        {"word": word, "recognized": recognized, "drawing": [[list(xs), list(ys)]]}
    )


def _fixture_root(tmp_path, per_class: int = 6):
    for word in ("cat", "sun"):
        lines = []
        for i in range(per_class):
            lines.append(_line(word, [0, 100 + i, 255], [0, 50, 200]))
        lines.append(_line(word, [0, 10], [0, 10], recognized=False))  # deve ser filtrada
        (tmp_path / f"{word}.ndjson").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_parse_line():
    word, rec, ink = parse_quickdraw_line(_line("cat", [0, 128, 255], [0, 64, 128]))
    assert word == "cat" and rec
    assert len(ink.strokes) == 1
    assert len(ink.strokes[0].points) == 3
    assert ink.strokes[0].points[1].x == 128.0


def test_prepare_sketch_resamples_to_uniform_density():
    _, _, ink = parse_quickdraw_line(_line("cat", [0, 255], [0, 0]))  # 1 segmento longo
    feats = prepare_sketch(ink, max_points=512, resample_step=0.025)
    # linha reta de comprimento ~2 (normalizada) → ~80 pontos, não 2
    assert 60 < len(feats) < 120
    assert all(len(f) == 6 for f in feats)


def test_dataset_split_is_deterministic_and_disjoint(tmp_path):
    root = _fixture_root(tmp_path, per_class=6)
    kw = dict(
        categories=["cat", "sun"], train_per_class=4, val_per_class=2, resample_step=None
    )
    train = QuickDrawDataset(root, split="train", **kw)
    val = QuickDrawDataset(root, split="val", **kw)
    assert len(train) == 8  # 4 por classe
    assert len(val) == 4  # 2 por classe
    # 'recognized: false' nunca entra
    assert all('"recognized": false' not in s for s, _ in train.samples + val.samples)
    # val = primeiras linhas; train = seguintes (disjuntos)
    assert set(train.samples).isdisjoint(set(val.samples))


def test_getitem_and_collate_shapes(tmp_path):
    root = _fixture_root(tmp_path)
    ds = QuickDrawDataset(
        root, categories=["cat", "sun"], split="train",
        train_per_class=3, val_per_class=1, resample_step=0.05,
    )
    feats, label = ds[0]
    assert feats.ndim == 2 and feats.shape[1] == 6
    assert label.item() == 0  # "cat" é a classe 0 (ordem das categorias)

    batch = SketchCollate()([ds[0], ds[-1]])
    assert batch["src"].shape[0] == 2 and batch["src"].shape[2] == 6
    assert batch["label"].tolist() == [0, 1]
    assert batch["src_key_padding_mask"].shape == batch["src"].shape[:2]
