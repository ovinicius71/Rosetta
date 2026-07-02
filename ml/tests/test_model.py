"""Testes do modelo (encoder + LatexHead). Exigem torch; pulados se ausente.

Rodar: uv run pytest ml/tests/test_model.py
"""

import pytest

torch = pytest.importorskip("torch")

from hmer_ml.data.dataset import Collate  # noqa: E402
from hmer_ml.model import build_model  # noqa: E402
from hmer_ml.utils.config import Config  # noqa: E402


def _tiny_cfg(vocab=20):
    return Config(
        {
            "model": {
                "head": "latex",
                "d_model": 32,
                "in_features": 6,
                "encoder": {"type": "bigru", "hidden": 32, "layers": 1, "dropout": 0.0},
                "decoder": {"layers": 2, "heads": 4, "ff": 64, "dropout": 0.0},
            }
        }
    )


def _fake_batch(b=2, t=7, l=5, pad_id=0):
    collate = Collate(pad_id)
    samples = []
    for i in range(b):
        feats = torch.randn(t - i, 6)  # comprimentos diferentes p/ exercitar padding
        toks = torch.randint(1, 19, (l,))
        samples.append((feats, toks))
    return collate(samples)


def test_forward_logits_shape():
    """forward (teacher forcing) devolve logits [B, L-1, vocab]."""
    cfg = _tiny_cfg(vocab=20)
    model = build_model(cfg, vocab_size=20, pad_id=0, bos_id=1, eos_id=2)
    batch = _fake_batch(b=2, l=5)
    logits = model(batch)
    assert logits.shape[0] == 2
    assert logits.shape[1] == batch["tgt"].size(1) - 1
    assert logits.shape[2] == 20


def test_greedy_decode_runs_and_stops():
    """greedy_decode devolve uma lista de ids por exemplo, sem <bos>, cortando em <eos>."""
    cfg = _tiny_cfg(vocab=20)
    model = build_model(cfg, vocab_size=20, pad_id=0, bos_id=1, eos_id=2)
    batch = _fake_batch(b=2, l=5)
    out = model.greedy_decode(batch, max_len=10)
    assert len(out) == 2
    assert all(2 not in seq and 1 not in seq[:1] for seq in out)  # sem eos; bos removido


def test_beam_search_beam1_matches_greedy():
    """Com beam_size=1 (sem empates patológicos), beam search == greedy decode."""
    from hmer_ml.infer import beam_search

    torch.manual_seed(0)
    cfg = _tiny_cfg(vocab=20)
    model = build_model(cfg, vocab_size=20, pad_id=0, bos_id=1, eos_id=2)
    model.eval()
    batch = _fake_batch(b=1, l=5)
    greedy = model.greedy_decode(batch, max_len=8)[0]
    memory, mmask = model.encode(batch)
    beam = beam_search(model.head, memory, mmask, bos_id=1, eos_id=2,
                       beam_size=1, max_len=8)
    assert beam == greedy


def test_beam_search_returns_valid_ids():
    from hmer_ml.infer import beam_search

    torch.manual_seed(1)
    cfg = _tiny_cfg(vocab=20)
    model = build_model(cfg, vocab_size=20, pad_id=0, bos_id=1, eos_id=2)
    model.eval()
    batch = _fake_batch(b=1, l=5)
    memory, mmask = model.encode(batch)
    out = beam_search(model.head, memory, mmask, bos_id=1, eos_id=2,
                      beam_size=4, max_len=8)
    assert isinstance(out, list)
    assert 2 not in out and all(0 <= t < 20 for t in out)  # sem eos, ids válidos


def test_encoder_is_task_agnostic():
    """ADR 0006: o encoder não pode depender do tokenizer/vocabulário nem da cabeça.

    Verifica acoplamento estrutural (imports e assinatura), não a prosa dos comentários:
      - o módulo do encoder não importa o tokenizer nem as heads;
      - build_encoder(cfg) constrói sem receber vocab_size/pad_id/bos/eos.
    """
    import inspect

    from hmer_ml.model import build_encoder
    from hmer_ml.model import encoder as enc_mod

    src = inspect.getsource(enc_mod)
    assert "import" in src
    assert "tokenizer" not in src, "encoder não deve importar/conhecer o tokenizer"
    assert ".heads" not in src and "LatexHead" not in src, "encoder não deve conhecer a cabeça"

    # build_encoder só depende da config — nada específico da tarefa na assinatura
    params = set(inspect.signature(build_encoder).parameters) - {"cfg"}
    assert params == set(), f"build_encoder acoplado à tarefa: {params}"
