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
- [x] **Treino no CROHME completo** (`configs/crohme.yaml`: 8.901 amostras, 60 épocas,
      batch efetivo 32, AMP, bucketing) — 1ª rodada overfitou (77% train vs 5% valid,
      CER 1.41): as features dependiam da densidade da caneta.
- [x] **Retreino com resample fixo** (`checkpoints/crohme_rs`, `resample_step: 0.004`,
      treino = inferência): no valid (986, beam 4) **CER 1.41 → 0.81**; exact 5.2%.
      Já é o checkpoint servido pela API (default do `serve_api.ps1`).
- [x] **Bug do resample corrigido** (2026-07-04): `carry` não acumulava entre segmentos
      curtos — tinta mais densa que o passo emitia MENOS pontos, quebrando a invariância
      prometida. Com o fix, as features são idênticas em qualquer densidade (teste de
      regressão em `test_smoke.py`) e as predições ×3-denso batem 15/15 com as originais.
      O checkpoint (treinado com o resample antigo) quase não perde: CER 0.815/exact 4.7%.
- [ ] Avaliar no test (1.199); comparar com literatura.
- [ ] Retreino limpo com o resample corrigido e `resample_step: 0.008` (pós-fix, o passo
      0.004 estoura `max_points` em 25% das amostras; 0.008 → p97=1085 e treino ~2× mais
      rápido). Ganho esperado: pequeno — o salto de verdade vem do MathWriting.
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

## Fase 3.5 — Xournal++ como interface principal  ← *concluída (2026-07-03)*
- [x] **Segmentação de contas** em página livre (`ml/segment.py`): detecta "=" por
      geometria (par de barras), agrupa a expressão da linha de escrita (frações e
      expoentes entram; texto de outras linhas/colunas não). 11 testes.
- [x] **Resultado como tinta** (`api/inkfont.py`): fonte vetorial Hershey Simplex
      (domínio público, gerada por `scripts/gen_inkfont_glyphs.py`) — o valor é
      *desenhado* após o "=", estilo iPad Math Notes.
- [x] `POST /page/process`: traços da página → contas pendentes → Recognizer → SymPy →
      polilinhas prontas. Cor fixa (laranja) marca resultado = idempotência (rodar de
      novo não duplica). Erros por conta são isolados. 7 testes (+4 inkfont).
- [x] **Plugin Lua** (`xournalpp-plugin/rosetta`): Ctrl+M → `getStrokes("layer")` →
      curl → `addStrokes` (undo agrupado). Instalação: `scripts/install_plugin.ps1`.
- [x] Verificado com tinta real do CROHME (`scripts/e2e_page_process.py`): segmentação
      acha o "=" em escrita real; pipeline completo responde.
- **Limitação conhecida:** a API Lua do Xournal++ (1.3.4) não tem eventos de traço —
  o gatilho é um atalho, não o ato de escrever. Gatilho em tempo real = patch C++
  upstream (futuro).
- **Mudança de rota (2026-07-03):** o plano Xournal++ foi **pausado** — a web virou o
  caderno (Fase 3.6). O plugin Lua continua no repo e funcional, mas não é o caminho.

## Fase 3.6 — O caderno web (interface principal)  ← *concluída (2026-07-03)*
- [x] `web/` remodelado: folha longa de anotações livres (cresce ao escrever), 3 canetas,
      2 pontas, borracha por traço, desfazer (Ctrl+Z), persistência em localStorage.
- [x] **Gatilho em tempo real** (o que o Xournal++ não permitia): pausou a caneta ~1,3s →
      a página vai a `/page/process` → resposta se desenha em tinta laranja animada.
      Toggle `auto` + botão `resolver =` na toolbar; status visível na própria toolbar.
- [x] Painel "contas resolvidas" (KaTeX) + design system Paper & Ink mantido (linha de
      margem, grade pontilhada). Componentes antigos (InkCanvas/LatexView) removidos.
- [x] API aquece o modelo no startup; proxy do Next com timeout de 180s (1ª inferência
      em CPU não derruba mais a chamada).
- [x] Verificado ao vivo no Chrome: "2+3=" → **5** laranja; "1+1=" resolvida pelo
      gatilho automático sem clique; reprocessar não duplica; reload preserva a página.
- **Saída atingida:** anotar livremente no browser e ver as contas se resolverem no
  meio das anotações.

## Fase 4 — Extensão para reconhecimento de desenhos  ← *v1 concluída (2026-07-03)*
- [x] **Cabeça de classificação** (`SketchClsHead`, mean-pool + linear) sobre o **mesmo
      encoder de tinta** — o branch já existia em `InkModel.forward` (ADR 0006 provado).
- [x] Dataset **QuickDraw** simplified: `scripts/download_quickdraw.py` (byte-range,
      21 categorias × ~14-25k desenhos, 169 MB) + `data/quickdraw.py` (split
      determinístico, `resample_step: 0.025` fixo — treino = inferência, lição da Fase 2;
      "heart" não existe no QuickDraw, virou "butterfly").
- [x] Config seleciona a cabeça (`configs/quickdraw.yaml`, `head: sketch_cls`) +
      `train_sketch.py` (CE, val acc por época, best.ckpt).
- [x] `infer_sketch.SketchRecognizer` + `POST /sketch/recognize` (SKETCH_CKPT, default
      `checkpoints/quickdraw/best.ckpt`; rótulo pt-BR desenhado em tinta sob o desenho).
- [x] Caderno: botão **`desenho?`** classifica o desenho recente (cluster espacial dos
      traços finais) + **múltiplas páginas** (◀ n/N ▶ + página, persistência v2).
- [x] Verificado no Chrome: círculo desenhado → rótulo "circulo" em tinta laranja +
      "círculo · 67%" (checkpoint parcial; treino completo em andamento).
- **Saída atingida:** o mesmo encoder serve matemática e "o que está sendo desenhado".
- Próximo: mais categorias; triagem automática math/texto/desenho na mesma passada.

## Fase 5 — Equações identificadas e plotadas (R² e R³)  ← *pipeline concluído (2026-07-03)*
- [x] **Segmentação com lado direito**: `group_expression` varre os dois lados do "="
      (`lhs_indices`/`rhs_indices`); "já resolvida" também vale para tinta ABAIXO da
      expressão (gráficos). Fecha o buraco em que o RHS de "x²+y²=4" era ignorado.
- [x] **Classificador** (`api/analyze.py`): cônicas R² (reta, circunferência, elipse,
      parábola, hipérbole — com rotação, via autovalores) e quádricas R³ (esfera,
      elipsoide, paraboloides, hiperboloides, cone, plano) + `y=f(x)` / `z=f(x,y)`
      via lambdify. Descrição em pt-BR com parâmetros (raio, centro, vértice…).
- [x] **Gráfico como tinta** (`api/inkplot.py`): eixos com setas/ticks/escala + curva
      paramétrica (R²) e wireframe isométrico (R³), tudo polilinhas Hershey/numpy —
      sem matplotlib, sem novo tipo de elemento no caderno.
- [x] **Integração**: `/page/process` distingue conta (RHS vazio → valor após o "=")
      de equação (RHS cheio → gráfico+descrição ABAIXO; 1 variável → solução abaixo).
      Painel mostra "x²+y²=4 → circunferencia · raio 2 · centro (0,0)". 101 testes.
- [x] **Retreino do CROHME com `resample_step: 0.004`** (`checkpoints/crohme_rs`) —
      concluído em 2026-07-04 (ver Fase 2): CER 1.41 → 0.81 no valid; servido pela API,
      com o resample manual removido de `page.py` (canonicalização é só do Recognizer).
- [x] **E2e com caligrafia real** (2026-07-04): "x²+y²=4" à mão → circunferência com
      eixos + descrição. Exigiu 4 correções: adoção de sobrescritos que pairam sobre o
      "=" (segment), beam 16 com escolha da 1ª hipótese *classificável* (equações),
      `analyze` aceitando qualquer par/trio de letras latinas como eixos, e teto de
      tamanho no texto-fallback. Limite honesto: o modelo ainda troca símbolos
      (y→p, 4→8 ⇒ "raio 2.83"); a forma sai certa, o número não — MathWriting resolve.
- Verificado visualmente no caderno: circunferência, esfera e sela renderizam bem.

## Explicitamente adiado (não implementar agora)
- Fusão multimodal (tinta + imagem renderizada).
- Refino da saída por LLM.
