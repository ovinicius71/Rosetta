# HMER — Online Handwritten Math Expression Recognition

Reconhece **matemática manuscrita** a partir da **tinta online** (trajetória da caneta,
não imagem rasterizada) e devolve a expressão em **LaTeX normalizado** — no espírito do
"Math Notes" do iPad. Quando fizer sentido, também calcula o resultado (SymPy).

> **Visão de longo prazo:** o *encoder de tinta* e o *esquema de tinta* são **agnósticos à
> tarefa**. Só a **cabeça de saída** muda entre matemática (decoder LaTeX) e reconhecimento
> de desenhos estilo QuickDraw (classificador). Ver [`docs/vision.md`](docs/vision.md).

## Monorepo

| Pacote       | Stack            | Papel                                                                        |
| ------------ | ---------------- | ---------------------------------------------------------------------------- |
| `ml/`      | Python + PyTorch | Dados (InkML→tensores), tokenizer, modelo, treino, avaliação, inferência |
| `api/`     | Python + FastAPI | `POST /recognize` (tinta→LaTeX), `POST /evaluate` (LaTeX→resultado)    |
| `web/`     | Next.js + canvas | Captura de tinta (PointerEvents), envio à API, render KaTeX                 |
| `schemas/` | JSON Schema      | **Contrato único da tinta** compartilhado por web/api/ml              |
| `docs/`    | Markdown         | Visão, datasets, ADRs, roadmap                                              |

## Decisões-chave

- Entrada = **tinta online** (traços/pontos), não imagem. Render→imagem é upgrade futuro.
- Saída = **LaTeX normalizado**.
- Treino **só local** numa ASUS TUF F16, VRAM ~6–8 GB → modelo compacto, AMP, grad accum, resume.
- Extensibilidade p/ desenhos é **requisito de arquitetura**.

Ver [`docs/roadmap.md`](docs/roadmap.md) para o plano faseado e [`docs/adr/`](docs/adr) para as decisões.

## Começando

```bash
# Python (ml + api) — workspace uv
uv sync

# Web
cd web && npm install && npm run dev
```

> **Datasets não são baixados neste momento.** Ver [`docs/datasets.md`](docs/datasets.md).

## Status

Fase 0 (scaffold). Nada treinado ainda. Stubs + TODOs marcam o trabalho seguinte.
