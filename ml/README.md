# ml/ — pacote de Machine Learning (Rosetta)

Pipeline de HMER online: **InkML → tinta → tensores → seq2seq → LaTeX**.

```
src/hmer_ml/
├── data/         # parsing InkML, esquema de tinta, dataset, augmentation
├── tokenizer/    # tokenizer de LaTeX custom (isolado)
├── model/        # encoder (compartilhado) + heads plugáveis + seq2seq
├── utils/        # config (YAML→dataclass), checkpoint/resume
├── train.py      # loop de treino (AMP, grad accum, resume)
├── evaluate.py   # CER + exact match, beam search
└── infer.py      # tinta → LaTeX (usado pela API)
configs/          # base.yaml, overfit_crohme.yaml, mathwriting.yaml
tests/            # smoke test do formato/shapes
```

## Comandos
```bash
# Fase 0 — dados
python -m hmer_ml.data.inkml   --stats data/crohme/train           # inspeciona InkML
python -m hmer_ml.data.build_vocab data/crohme/train --out artifacts/vocab.json

# Fase 1 — prova de overfit (dados sintéticos, sem download)
python -m hmer_ml.data.synth   --out data/synth --n 32             # gera InkML sintético
python -m hmer_ml.train        --config configs/overfit_synth.yaml # memoriza (loss→~0)
python -m hmer_ml.evaluate     --config configs/overfit_synth.yaml \
                               --ckpt checkpoints/overfit_synth/last.ckpt   # exact=1.0

# CROHME real: baixe (ver docs/datasets.md) e use configs/overfit_crohme.yaml
```
> Rode com `PYTHONPATH=ml/src` (ou instale o pacote via `uv sync`). Se treinar em GPU,
> instale a build CUDA do torch; a `+cpu` funciona mas não usa a GPU.

## Restrições de hardware (ASUS TUF F16, VRAM ~6–8 GB)
Modelo compacto por default, **AMP**, **acumulação de gradiente**, **bucketing/padding**
por comprimento, **checkpoint com resume**. Escalar via config, não via edição de código.
