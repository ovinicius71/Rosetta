"""Teste ponta a ponta manual de POST /page/process com tinta REAL do CROHME.

Procura amostras de treino cujo label contém '=', converte a tinta para o formato de
página do Xournal++ (x[]/y[] por traço) e envia à API local — exercitando segmentação
(achar o '=' em escrita real), reconhecimento e desenho do resultado.

Uso (API já no ar):  python scripts/e2e_page_process.py [n_amostras]
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml" / "src"))

from hmer_ml.data.inkml import iter_inkml  # noqa: E402

API = "http://127.0.0.1:8000/page/process"


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    found = 0
    for path, ink in iter_inkml(REPO / "data" / "crohme" / "train"):
        if not ink.label or "=" not in ink.label:
            continue
        found += 1
        page = {
            "strokes": [
                {"x": [p.x for p in s.points], "y": [p.y for p in s.points], "width": 1.41}
                for s in ink.strokes
            ]
        }
        resp = post(page)
        exprs = resp["expressions"]
        print(f"\n{Path(path).name}  label: {ink.label}")
        if not exprs:
            print("  (nenhum '=' detectado pela segmentacao)")
        for e in exprs:
            ok = "ok" if e["strokes"] else "sem tinta"
            print(f"  latex: {e['latex']!r}")
            print(f"  result: {e['result']!r}  erro: {e['error']!r}  [{ok}]")
        if found >= limit:
            break
    if not found:
        print("nenhuma amostra com '=' encontrada")


if __name__ == "__main__":
    main()
