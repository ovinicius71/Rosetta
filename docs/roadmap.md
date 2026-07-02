# Roadmap (faseado)

Cada fase é fatiável em várias sessões de treino (checkpoint + resume). Datasets pesados
ficam fora do git.

## Fase 0 — Scaffold + fundação de dados  ← *concluída (falta validar collate com torch)*
- [x] Scaffold do monorepo (ml/api/web/schemas/docs).
- [x] **Esquema de tinta** compartilhado (`schemas/ink.schema.json`) + exemplo.
- [x] Parsing **InkML → tinta** (`ml/data/inkml.py`) + features `[T,6]`, normalize,
      resample (`ml/data/ink.py`). Validado com InkML sintético.
- [x] **Tokenizer** de LaTeX custom: tokenize/build_vocab/encode/decode + especiais;
      CLI `hmer_ml.data.build_vocab`. Round-trip testado.
- [x] `Dataset` + `collate` + `LengthBucketSampler` (`ml/data/dataset.py`);
      `prepare_sample` testável sem torch. **`collate_fn` pende validação com torch.**
- **Saída atingida:** `python -m hmer_ml.data.inkml --stats` roda; `prepare_sample`
  produz features e tokens corretos. Próximo: instalar torch e conferir shapes do batch.

## Fase 1 — Overfit (provar o seq2seq ponta a ponta)  ← *concluída*
- [x] Encoder (BiGRU) + Transformer decoder (LatexHead) + teacher forcing.
- [x] Loop de treino: AMP, grad accum, grad clip, warmup, checkpoint/resume, logging.
- [x] Decode greedy (`infer.Recognizer`) + métricas CER/exact-match (`evaluate.py`).
- [x] Overfit em sintético (`data/synth`, 32 amostras): exact=1.0, cer=0.0.
- [x] Overfit em **CROHME real** (32 amostras, GPU RTX 5050, AMP): loss 5.03→0.0008,
      exact=1.0, cer=0.0 (greedy e beam=4).
- **Saída atingida:** o seq2seq aprende ponta a ponta em dados reais, na GPU.

## Fase 2 — Escalar (CROHME completo → MathWriting)  ← *em andamento*
- [x] **Augmentation** de tinta (`data/augment.py`): jitter, escala anisotrópica,
      rotação leve, dropout de pontos — classes picláveis, agnósticas à tarefa.
- [x] **Beam search** com length normalization (`infer.beam_search`), ligado ao
      Recognizer e ao `evaluate` (`--beam`).
- [x] Avaliação por split: `evaluate --root data/crohme/valid [--limit N] [--verbose]`.
- [x] Vocab por dataset (`artifacts/vocab_<dataset>.json`) — troca de dataset nunca
      reusa vocabulário errado.
- [~] **Treino no CROHME completo** (`configs/crohme.yaml`: 8.901 amostras, 60 épocas,
      batch efetivo 32, AMP, bucketing) — rodando na RTX 5050.
- [ ] Avaliar no valid (986) e test (1.199) com CER/exact match; comparar com literatura.
- [ ] Baixar MathWriting e treinar (`configs/mathwriting.yaml` pronto).
- **Saída:** métricas comparáveis à literatura de HMER online num split de validação.

## Fase 3 — Inferência servida + canvas + render  ← *concluída*
- [x] `POST /recognize` carrega o checkpoint via `HMER_CKPT`/`HMER_CONFIG` e devolve
      LaTeX (beam search do config; erros viram HTTP amigável, não 500).
- [x] Canvas Next.js → proxy `/api/*` → FastAPI → **render KaTeX** (CSS importado).
- [x] `POST /evaluate` com **SymPy**: aritmética exata, frações, raízes e resolução de
      equações (`2x+4=10` → `x = 3`); expressão inválida → erro amigável.
- [x] Verificado ponta a ponta: tinta real do CROHME via `:3000/api/recognize` →
      LaTeX exato do gold. 8/8 testes da API (incl. integração com modelo real).
- **Saída atingida:** desenhar no browser e ver LaTeX + resultado.
  Rodar: API `uvicorn hmer_api.main:app` (com HMER_CKPT) + `npm run dev` em web/.

## Fase 4 — Extensão para reconhecimento de desenhos
- [ ] Nova **cabeça de classificação** sobre o **mesmo encoder de tinta**.
- [ ] Dataset **QuickDraw** (também tinta online).
- [ ] Config seleciona a cabeça (`head: latex` | `head: sketch_cls`).
- **Saída:** o mesmo encoder serve matemática e "o que está sendo desenhado".

## Explicitamente adiado (não implementar agora)
- Fusão multimodal (tinta + imagem renderizada).
- Refino da saída por LLM.
