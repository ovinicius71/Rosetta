"""Avaliação: CER (character/symbol error rate) e exact match por expressão.

Alinhado à literatura de HMER online. O "CER" aqui é medido em **tokens de LaTeX**
(símbolos) — a granularidade que interessa em HMER — via distância de edição normalizada.
Ver docs/roadmap.md (Fase 2).

Uso:
  python -m hmer_ml.evaluate --config configs/crohme.yaml --ckpt checkpoints/crohme/last.ckpt \
      --root data/crohme/valid [--limit 200] [--beam 4]
"""

from __future__ import annotations

import argparse
import itertools

from .tokenizer import LatexTokenizer
from .utils.config import load_config


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Distância de Levenshtein entre sequências de tokens (DP O(len(a)*len(b)))."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def char_error_rate(pred: str, gold: str) -> float:
    """Edit distance em tokens de LaTeX, normalizada pelo nº de tokens do gold."""
    p = LatexTokenizer.tokenize(pred)
    g = LatexTokenizer.tokenize(gold)
    if not g:
        return 0.0 if not p else 1.0
    return _edit_distance(p, g) / len(g)


def exact_match(pred: str, gold: str) -> bool:
    """Igualdade após normalização (retokeniza e compara — ignora espaçamento)."""
    return LatexTokenizer.tokenize(pred) == LatexTokenizer.tokenize(gold)


def evaluate(config_path: str, ckpt: str, root: str | None = None,
             limit: int | None = None, beam_size: int | None = None,
             device: str | None = None, verbose: bool = False) -> dict:
    """Roda inferência sobre um diretório de InkML e agrega CER médio e exact-match rate."""
    import torch

    from .data.inkml import iter_inkml
    from .infer import Recognizer

    cfg = load_config(config_path)
    root = root or cfg.data.root
    if beam_size is None:
        beam_size = cfg.get("infer", {}).get("beam_size", 1) if hasattr(cfg, "get") else 1
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rec = Recognizer(ckpt, config_path=config_path, device=device)
    max_len = cfg.get("infer", {}).get("max_len", 256)

    n = exact = 0
    cer_sum = 0.0
    it = iter_inkml(root)
    if limit:
        it = itertools.islice(it, limit)
    for path, ink in it:
        if not ink.label:
            continue
        pred = rec.recognize(ink, beam_size=beam_size, max_len=max_len)
        n += 1
        ok = exact_match(pred, ink.label)
        exact += int(ok)
        cer_sum += char_error_rate(pred, ink.label)
        if verbose and not ok:
            print(f"[eval] {path.name}\n  gold: {ink.label}\n  pred: {pred}")
    return {
        "n": n,
        "exact_match": round(exact / n, 4) if n else 0.0,
        "cer": round(cer_sum / n, 4) if n else 0.0,
        "beam_size": beam_size,
        "device": device,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação HMER (CER, exact match)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", help="diretório de InkML a avaliar (default: data.root do config)")
    ap.add_argument("--limit", type=int, help="avalia só as N primeiras amostras")
    ap.add_argument("--beam", type=int, help="beam size (default: infer.beam_size do config)")
    ap.add_argument("--verbose", action="store_true", help="imprime os erros (gold vs pred)")
    args = ap.parse_args()
    metrics = evaluate(args.config, args.ckpt, root=args.root, limit=args.limit,
                       beam_size=args.beam, verbose=args.verbose)
    print(metrics)


if __name__ == "__main__":
    main()
