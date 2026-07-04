"""Gera api/src/hmer_api/_hershey.py a partir de scripts/futural.jhf (Hershey Simplex).

Fonte: https://github.com/kamalmostafa/hershey-fonts (dados Hershey, domínio público).
Formato .jhf por glifo: cols 0-4 id, cols 5-7 nº de vértices (incluindo o par de margens),
depois pares de chars codificados como offset de 'R' ('R'=0). O par " R" é pen-up.

Rodar da raiz do repo: python scripts/gen_inkfont_glyphs.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "futural.jhf"
OUT = REPO / "api" / "src" / "hmer_api" / "_hershey.py"


def parse_jhf(path: Path) -> dict[str, tuple[int, int, list[list[tuple[int, int]]]]]:
    glyphs: dict[str, tuple[int, int, list[list[tuple[int, int]]]]] = {}
    lines = path.read_text().splitlines()
    for i, raw in enumerate(lines):
        code = chr(32 + i)  # futural.jhf = ASCII printável em ordem, a partir do espaço
        nverts = int(raw[5:8])
        data = raw[8:]
        assert len(data) == 2 * nverts, f"glifo {code!r}: esperado {2*nverts} chars"
        left = ord(data[0]) - ord("R")
        right = ord(data[1]) - ord("R")
        strokes: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        for k in range(1, nverts):
            pair = data[2 * k : 2 * k + 2]
            if pair == " R":  # pen-up
                if current:
                    strokes.append(current)
                current = []
                continue
            x = ord(pair[0]) - ord("R")
            y = ord(pair[1]) - ord("R")
            current.append((x, y))
        if current:
            strokes.append(current)
        glyphs[code] = (left, right, strokes)
    return glyphs


def main() -> None:
    glyphs = parse_jhf(SRC)

    # sanity: orientação do eixo y (deve crescer para baixo, como na página)
    for probe in ("0", "L", "7"):
        _, _, st = glyphs[probe]
        ys = [y for s in st for _, y in s]
        print(f"glifo {probe!r}: y em [{min(ys)}, {max(ys)}], {len(st)} tracos")

    body = ["# GERADO por scripts/gen_inkfont_glyphs.py — não editar à mão.",
            '"""Fonte vetorial Hershey Simplex (futural), domínio público.',
            "",
            "GLYPHS[char] = (margem_esq, margem_dir, traços); traço = lista de (x, y) em",
            "unidades da fonte, y crescendo para baixo, origem no centro da linha ('R').",
            '"""',
            "",
            "GLYPHS: dict[str, tuple[int, int, list[list[tuple[int, int]]]]] = {"]
    for ch, (left, right, strokes) in glyphs.items():
        body.append(f"    {ch!r}: ({left}, {right}, {strokes!r}),")
    body.append("}")
    OUT.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"escrito {OUT} ({len(glyphs)} glifos)")


if __name__ == "__main__":
    main()
