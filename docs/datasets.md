# Datasets

> **NÃO baixar nada nesta fase.** Este documento só registra fontes oficiais, licenças,
> formato e instruções. Dados ficam **fora do git** (ver `.gitignore`) em `data/`.

## Layout local esperado (fora do git)
```
data/
├── crohme/            # secundário / benchmark
│   ├── train/  *.inkml
│   └── test/   *.inkml   (2014, 2016, 2019)
└── mathwriting/       # primário
    ├── train/  *.inkml
    ├── valid/  *.inkml
    ├── test/   *.inkml
    └── synthetic/ *.inkml
```

## 1. MathWriting (Google, 2024) — **primário**
- Maior dataset online de HMER: **~230k amostras humanas + ~400k sintéticas**.
- Formato **InkML**; ground truth em **LaTeX normalizado**.
- **244 símbolos matemáticos + 10 tokens sintáticos**; inclui símbolos isolados.
- Cobre a faixa de cálculo desejada.
- Licença: **Creative Commons** (conferir termos exatos no release oficial ao baixar).
- Link oficial: https://github.com/google-research/google-research/tree/master/mathwriting
  (dataset hospedado em Google Cloud Storage — ver instruções no repositório).

## 2. CROHME (2011–2019) — **secundário / benchmark**
- **~8.836 amostras de treino**; InkML com trajetória + GT em **LaTeX e MathML**.
- Conjuntos de teste **2014 / 2016 / 2019** (986 / 1.147 / 1.199 expressões).
- Menor → ideal para **validar o pipeline rápido** antes de escalar p/ MathWriting.
- Fontes (confirmadas jul/2026):
  - **Kaggle** (mais direto, InkML pronto): https://www.kaggle.com/datasets/ntcuong2103/crohme2019
  - **Página oficial RIT / CROHME 2019:** https://www.cs.rit.edu/~crohme2019/dataANDtools.html
  - Catálogo TC10/TC11: https://tc101-demo.github.io/datasets/ICDAR2019-CROHME-TDF_1/
- Estrutura do InkML: label da **expressão** em `<annotation type="truth">` (filho direto de
  `<ink>`); os `<traceGroup>` contêm truths **por símbolo** — nosso parser ignora esses e
  pega só o da expressão (ver `data/inkml.py::_extract_label`).

## Formato InkML (resumo)
XML com traços em `<trace>` (listas de `x y [t]` separados por vírgula) e anotações em
`<annotation type="truth">...</annotation>` com o LaTeX. O parser vive em
`ml/src/hmer_ml/data/inkml.py` e converte InkML → nosso **esquema de tinta compartilhado**
(`schemas/ink.schema.json`), garantindo que treino e inferência usem a **mesma
representação**.

## Instruções de download (quando chegar a hora — Fase 2)
1. CROHME primeiro (pequeno): baixar, extrair em `data/crohme/`.
2. MathWriting depois: seguir instruções do repositório oficial (GCS), extrair em
   `data/mathwriting/`.
3. Rodar `uv run python -m hmer_ml.data.inkml --stats <dir>` para conferir contagens.
4. Nunca commitar `data/`.
