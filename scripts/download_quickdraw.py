"""Baixa um subset do Google QuickDraw (formato "simplified" .ndjson) para a Fase 4.

Cada categoria é um .ndjson público de centenas de MB; só precisamos de alguns milhares
de amostras, então baixamos por **byte-range** (primeiros N MB) e descartamos a última
linha (potencialmente truncada). Fonte pública:
https://storage.googleapis.com/quickdraw_dataset/full/simplified/<categoria>.ndjson

Uso (da raiz do repo):  python scripts/download_quickdraw.py [--mb 8]
Saída: data/quickdraw/<categoria>.ndjson (só linhas completas)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "quickdraw"
BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"

# 21 categorias comuns e visualmente distintas (nomes oficiais do QuickDraw).
# A ordem AQUI é a ordem das classes do modelo — não reordenar depois de treinar.
CATEGORIES = [
    "cat", "dog", "house", "tree", "sun", "car", "flower", "fish", "bird", "star",
    "clock", "bicycle", "airplane", "sailboat", "cup", "chair", "apple", "moon",
    "umbrella", "butterfly", "circle",  # "heart" não existe no QuickDraw (404)
]


def download_category(name: str, max_bytes: int) -> int:
    """Baixa os primeiros max_bytes da categoria; retorna nº de linhas completas."""
    dst = OUT / f"{name}.ndjson"
    url = f"{BASE}/{name.replace(' ', '%20')}.ndjson"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    # descarta a última linha (quase certamente cortada pelo range)
    body = raw[: raw.rfind(b"\n") + 1]
    dst.write_bytes(body)
    return body.count(b"\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=8, help="MB por categoria (default 8)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in CATEGORIES:
        dst = OUT / f"{name}.ndjson"
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[qd] {name}: já existe ({dst.stat().st_size // 1024} KB) — pulando")
            continue
        n = download_category(name, args.mb * 1024 * 1024)
        print(f"[qd] {name}: {n} desenhos")
    print(f"[qd] concluído em {OUT}")


if __name__ == "__main__":
    main()
