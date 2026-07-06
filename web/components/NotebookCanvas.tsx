"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, processPage, recognizeSketch } from "@/lib/api";
import {
  PAGE_MIN_HEIGHT,
  PENS,
  PEN_WIDTHS,
  RESULT_COLOR,
  clearNotebook,
  emptyPage,
  intToCss,
  loadNotebook,
  saveNotebook,
  type NotebookPage,
  type NoteStroke,
  type SolvedConta,
} from "@/lib/notebook";

interface Props {
  /** Contas resolvidas da página ATUAL, sempre que mudarem (para o painel). */
  onSolvedChange: (solved: SolvedConta[]) => void;
}

type Tool = "pen" | "eraser";
type Status =
  | { kind: "idle"; msg?: string }
  | { kind: "busy"; msg: string }
  | { kind: "error"; msg: string };

const PAGE_GROW = 700; // quanto a folha cresce ao escrever perto do fim
const PAGE_MAX_H = 8000;
const SOLVE_DEBOUNCE_MS = 1300; // pausa de escrita que dispara a detecção de contas
const ERASER_RADIUS = 12;
const SKETCH_GAP = 70; // px: traços a menos disso um do outro formam "o desenho recente"

type UndoAction =
  | { kind: "add"; count: number } // traços adicionados no fim (caneta=1, resultado=N)
  | { kind: "erase"; items: { stroke: NoteStroke; index: number }[] };

type BBox = [number, number, number, number];

function bboxOf(s: NoteStroke): BBox {
  return [Math.min(...s.x), Math.min(...s.y), Math.max(...s.x), Math.max(...s.y)];
}

function boxGap(a: BBox, b: BBox): number {
  const dx = Math.max(0, Math.max(a[0] - b[2], b[0] - a[2]));
  const dy = Math.max(0, Math.max(a[1] - b[3], b[1] - a[3]));
  return Math.max(dx, dy);
}

/**
 * O caderno: páginas de escrita livre (anotações + matemática + desenhos misturados).
 *
 * A tinta é a fonte da verdade; o canvas é só desenho (mesmo princípio do ADR 0001).
 * Ao pausar a escrita, a página vai a POST /page/process — contas terminadas em "="
 * voltam resolvidas como tinta laranja que se desenha sozinha. O botão "desenho?" manda
 * os traços recentes a POST /sketch/recognize e o rótulo é escrito sob o desenho.
 */
export default function NotebookCanvas({ onSolvedChange }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pagesRef = useRef<NotebookPage[]>([emptyPage()]);
  const curRef = useRef(0);
  const strokesRef = useRef<NoteStroke[]>(pagesRef.current[0].strokes);
  const solvedRef = useRef<SolvedConta[]>(pagesRef.current[0].solved);
  const undoRef = useRef<UndoAction[]>([]);
  const drawingRef = useRef(false);
  const erasedInDragRef = useRef<{ stroke: NoteStroke; index: number }[]>([]);
  const solveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const busyRef = useRef(false);
  const solveAgainRef = useRef(false);
  const lastSentRef = useRef("");
  // animação "a tinta se escreve": progresso derivado do relógio (sobrevive a aba em
  // background e a lotes concorrentes — cada traço tem seu próprio start/dur)
  const animRef = useRef<Map<NoteStroke, { start: number; dur: number }>>(new Map());
  const rafRef = useRef(0);
  const animRunningRef = useRef(false);

  const [tool, setTool] = useState<Tool>("pen");
  const [penColor, setPenColor] = useState<number>(PENS[0].color);
  const [penWidth, setPenWidth] = useState<number>(PEN_WIDTHS[0].width);
  const [auto, setAuto] = useState(true);
  const [pageH, setPageH] = useState(PAGE_MIN_HEIGHT);
  const [pageInfo, setPageInfo] = useState({ cur: 0, total: 1 });
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [strokeCount, setStrokeCount] = useState(0);
  const [confirmClear, setConfirmClear] = useState(false);

  const ctx = () => canvasRef.current!.getContext("2d")!;

  // ── desenho ────────────────────────────────────────────────────────────────

  const paintStroke = useCallback((s: NoteStroke, upTo?: number) => {
    const c = ctx();
    const n = upTo === undefined ? s.x.length : Math.max(1, Math.min(upTo, s.x.length));
    c.strokeStyle = intToCss(s.color);
    c.lineWidth = s.width;
    c.lineCap = "round";
    c.lineJoin = "round";
    c.beginPath();
    c.moveTo(s.x[0], s.y[0]);
    for (let i = 1; i < n; i++) c.lineTo(s.x[i], s.y[i]);
    if (n === 1) c.lineTo(s.x[0] + 0.1, s.y[0]);
    c.stroke();
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const c = ctx();
    const dpr = window.devicePixelRatio || 1;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    const anims = animRef.current;
    const now = performance.now();
    for (const s of strokesRef.current) {
      const meta = anims.get(s);
      if (meta) {
        const p = Math.min(1, Math.max(0, (now - meta.start) / meta.dur));
        paintStroke(s, Math.max(1, Math.ceil(p * s.x.length)));
        if (p >= 1) anims.delete(s);
      } else {
        paintStroke(s);
      }
    }
  }, [paintStroke]);

  const resize = useCallback(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(wrap.clientWidth * dpr);
    canvas.height = Math.round(pageH * dpr);
    canvas.style.height = `${pageH}px`;
    redraw();
  }, [pageH, redraw]);

  useEffect(() => {
    resize();
    const ro = new ResizeObserver(resize);
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [resize]);

  // ── páginas e persistência ─────────────────────────────────────────────────

  const persist = useCallback(() => {
    const page = pagesRef.current[curRef.current];
    page.pageHeight = pageH;
    page.solved = solvedRef.current;
    saveNotebook({ version: 2, pages: pagesRef.current, current: curRef.current });
  }, [pageH]);

  /** Aponta os refs para a página `i` e sincroniza a UI. */
  const bindPage = useCallback(
    (i: number) => {
      curRef.current = i;
      const page = pagesRef.current[i];
      strokesRef.current = page.strokes;
      solvedRef.current = page.solved;
      undoRef.current = [];
      animRef.current.clear();
      lastSentRef.current = "";
      setStrokeCount(page.strokes.length);
      setPageInfo({ cur: i, total: pagesRef.current.length });
      setConfirmClear(false);
      setStatus({ kind: "idle" });
      onSolvedChange([...page.solved]);
      setPageH(Math.max(PAGE_MIN_HEIGHT, page.pageHeight || PAGE_MIN_HEIGHT));
      requestAnimationFrame(() => redraw()); // cobre o caso de pageH não mudar
    },
    [onSolvedChange, redraw]
  );

  useEffect(() => {
    const saved = loadNotebook();
    if (saved) {
      pagesRef.current = saved.pages;
      bindPage(Math.min(saved.current, saved.pages.length - 1));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const switchPage = useCallback(
    (i: number) => {
      if (i < 0 || i >= pagesRef.current.length || i === curRef.current) return;
      persist();
      bindPage(i);
    },
    [persist, bindPage]
  );

  const newPage = useCallback(() => {
    persist();
    pagesRef.current.push(emptyPage());
    bindPage(pagesRef.current.length - 1);
    persist();
  }, [persist, bindPage]);

  // ── resolver contas ────────────────────────────────────────────────────────

  const solve = useCallback(async () => {
    if (busyRef.current) {
      solveAgainRef.current = true;
      return;
    }
    const strokes = strokesRef.current;
    if (!strokes.some((s) => !s.result)) return;
    // nada mudou desde a última chamada → não gasta o modelo
    const signature = `${strokes.length}:${strokes.reduce((a, s) => a + s.x.length, 0)}`;
    if (signature === lastSentRef.current) return;

    busyRef.current = true;
    setStatus({ kind: "busy", msg: "procurando contas…" });
    try {
      const { expressions } = await processPage(
        strokes.map((s) => ({ x: s.x, y: s.y, color: s.color, width: s.width }))
      );
      lastSentRef.current = signature;

      const newStrokes: NoteStroke[] = [];
      const newSolved: SolvedConta[] = [];
      let failed = 0;
      for (const e of expressions) {
        const resultText = e.result ?? e.description; // valor OU descrição da curva
        if (e.strokes.length > 0 && resultText) {
          for (const s of e.strokes) {
            newStrokes.push({
              x: s.x,
              y: s.y,
              color: s.color ?? RESULT_COLOR,
              width: s.width ?? 2,
              result: true,
            });
          }
          newSolved.push({
            latex: e.latex,
            result: resultText,
            at: Date.now(),
            eq: e.description != null,
          });
        } else {
          failed++;
        }
      }

      if (newStrokes.length > 0) {
        commitResultStrokes(newStrokes);
        solvedRef.current = [...solvedRef.current, ...newSolved];
        onSolvedChange([...solvedRef.current]);
        persist();
      }
      setStatus(
        failed > 0
          ? { kind: "idle", msg: `${failed} conta${failed > 1 ? "s" : ""} ilegível — tente reescrever` }
          : { kind: "idle" }
      );
    } catch (err) {
      if (err instanceof ApiError && err.status === 501) {
        setStatus({ kind: "error", msg: "modelo não carregado — defina HMER_CKPT na API" });
      } else {
        setStatus({ kind: "error", msg: "api fora do ar? rode scripts/serve_api.ps1" });
      }
    } finally {
      busyRef.current = false;
      if (solveAgainRef.current) {
        solveAgainRef.current = false;
        scheduleSolve();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onSolvedChange, persist]);

  const scheduleSolve = useCallback(() => {
    if (solveTimerRef.current) clearTimeout(solveTimerRef.current);
    solveTimerRef.current = setTimeout(() => void solve(), SOLVE_DEBOUNCE_MS);
  }, [solve]);

  // ── reconhecer desenho (Fase 4) ────────────────────────────────────────────

  const sketchGuess = useCallback(async () => {
    if (busyRef.current) return;
    // "o desenho recente": traços do usuário no fim da lista, espacialmente encadeados
    const strokes = strokesRef.current;
    const cluster: NoteStroke[] = [];
    let bbox: BBox | null = null;
    for (let i = strokes.length - 1; i >= 0; i--) {
      const s = strokes[i];
      if (s.result) continue;
      const sb = bboxOf(s);
      if (bbox && boxGap(bbox, sb) > SKETCH_GAP) break;
      cluster.push(s);
      bbox = bbox
        ? [
            Math.min(bbox[0], sb[0]),
            Math.min(bbox[1], sb[1]),
            Math.max(bbox[2], sb[2]),
            Math.max(bbox[3], sb[3]),
          ]
        : sb;
    }
    if (cluster.length === 0) {
      setStatus({ kind: "idle", msg: "desenhe algo primeiro" });
      return;
    }

    busyRef.current = true;
    setStatus({ kind: "busy", msg: "olhando o desenho…" });
    try {
      const resp = await recognizeSketch(
        cluster.map((s) => ({ x: s.x, y: s.y, color: s.color, width: s.width }))
      );
      const label: NoteStroke[] = resp.strokes.map((s) => ({
        x: s.x,
        y: s.y,
        color: s.color ?? RESULT_COLOR,
        width: s.width ?? 2,
        result: true,
      }));
      if (label.length > 0) {
        commitResultStrokes(label);
        persist();
      }
      setStatus({
        kind: "idle",
        msg: `${resp.label_pt} · ${Math.round(resp.confidence * 100)}%`,
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 501) {
        setStatus({ kind: "error", msg: "classificador de desenhos ainda não treinado" });
      } else {
        setStatus({ kind: "error", msg: "api fora do ar? rode scripts/serve_api.ps1" });
      }
    } finally {
      busyRef.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persist]);

  /** Adiciona tinta de resultado com animação de "se escrevendo" + undo agrupado. */
  const commitResultStrokes = (batch: NoteStroke[]) => {
    const perStroke = 220; // ms
    const now = performance.now();
    batch.forEach((s, i) =>
      animRef.current.set(s, { start: now + i * perStroke * 0.7, dur: perStroke })
    );
    strokesRef.current.push(...batch);
    undoRef.current.push({ kind: "add", count: batch.length });
    setStrokeCount(strokesRef.current.length);
    lastSentRef.current = ""; // a tinta nova muda a página
    ensureAnimLoop();
  };

  /** Loop único de animação: roda enquanto houver traço com progresso < 1. */
  const ensureAnimLoop = useCallback(() => {
    if (animRunningRef.current) return;
    animRunningRef.current = true;
    const tick = () => {
      redraw(); // redraw calcula o progresso pelo relógio e remove os concluídos
      if (animRef.current.size > 0) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        animRunningRef.current = false;
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [redraw]);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  // ── entrada (caneta / borracha) ────────────────────────────────────────────

  const toXY = (e: React.PointerEvent): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  const eraseAt = (x: number, y: number) => {
    const r2 = ERASER_RADIUS * ERASER_RADIUS;
    const strokes = strokesRef.current;
    for (let i = strokes.length - 1; i >= 0; i--) {
      const s = strokes[i];
      for (let k = 0; k < s.x.length; k++) {
        const dx = s.x[k] - x;
        const dy = s.y[k] - y;
        if (dx * dx + dy * dy <= r2) {
          erasedInDragRef.current.push({ stroke: s, index: i });
          strokes.splice(i, 1);
          break;
        }
      }
    }
  };

  const onPointerDown = (e: React.PointerEvent) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* pointer já finalizado/sintético — desenhar mesmo assim */
    }
    drawingRef.current = true;
    setConfirmClear(false);
    if (tool === "eraser") {
      erasedInDragRef.current = [];
      const [x, y] = toXY(e);
      eraseAt(x, y);
      redraw();
      return;
    }
    const [x, y] = toXY(e);
    strokesRef.current.push({ x: [x], y: [y], color: penColor, width: penWidth });
    setStrokeCount(strokesRef.current.length);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drawingRef.current) return;
    if (tool === "eraser") {
      const [x, y] = toXY(e);
      eraseAt(x, y);
      redraw();
      return;
    }
    const s = strokesRef.current.at(-1)!;
    // getCoalescedEvents dá a trajetória completa em telas de alta taxa; pode vir vazio
    // (browsers antigos / eventos sintéticos) — nesse caso usa o próprio evento.
    const coalesced = e.nativeEvent.getCoalescedEvents?.() ?? [];
    const events = coalesced.length > 0 ? coalesced : [e.nativeEvent];
    const rect = canvasRef.current!.getBoundingClientRect();
    const c = ctx();
    c.strokeStyle = intToCss(s.color);
    c.lineWidth = s.width;
    c.lineCap = "round";
    c.beginPath();
    c.moveTo(s.x.at(-1)!, s.y.at(-1)!);
    for (const ev of events) {
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      s.x.push(x);
      s.y.push(y);
      c.lineTo(x, y);
    }
    c.stroke();
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* idem */
    }

    if (tool === "eraser") {
      if (erasedInDragRef.current.length > 0) {
        undoRef.current.push({ kind: "erase", items: erasedInDragRef.current });
        erasedInDragRef.current = [];
        setStrokeCount(strokesRef.current.length);
        lastSentRef.current = "";
        persist();
        if (auto) scheduleSolve();
      }
      return;
    }

    const s = strokesRef.current.at(-1);
    if (s) {
      undoRef.current.push({ kind: "add", count: 1 });
      // a folha cresce quando a escrita se aproxima do fim
      const maxY = Math.max(...s.y);
      if (maxY > pageH - 250 && pageH < PAGE_MAX_H) {
        setPageH(Math.min(PAGE_MAX_H, pageH + PAGE_GROW));
      }
    }
    persist();
    if (auto) scheduleSolve();
  };

  // ── ações da toolbar ───────────────────────────────────────────────────────

  const undo = useCallback(() => {
    const action = undoRef.current.pop();
    if (!action) return;
    if (action.kind === "add") {
      strokesRef.current.splice(-action.count, action.count);
    } else {
      // devolve os traços apagados às posições originais (ordem crescente)
      const items = [...action.items].sort((a, b) => a.index - b.index);
      for (const { stroke, index } of items) {
        strokesRef.current.splice(Math.min(index, strokesRef.current.length), 0, stroke);
      }
    }
    setStrokeCount(strokesRef.current.length);
    lastSentRef.current = "";
    redraw();
    persist();
  }, [redraw, persist]);

  const clearPage = () => {
    if (!confirmClear) {
      setConfirmClear(true);
      return;
    }
    strokesRef.current.length = 0; // mantém o alias com pages[cur].strokes
    solvedRef.current = [];
    undoRef.current = [];
    animRef.current.clear();
    lastSentRef.current = "";
    setStrokeCount(0);
    setConfirmClear(false);
    setStatus({ kind: "idle" });
    setPageH(PAGE_MIN_HEIGHT);
    onSolvedChange([]);
    if (pagesRef.current.length === 1) clearNotebook();
    else persist();
    redraw();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo]);

  const hasInk = strokeCount > 0;

  return (
    <div className="notebook-shell">
      <div className="notebook-toolbar" role="toolbar" aria-label="ferramentas do caderno">
        <div className="tool-group" aria-label="canetas">
          {PENS.map((p) => (
            <button
              key={p.name}
              title={`caneta ${p.name}`}
              className={`swatch ${tool === "pen" && penColor === p.color ? "active" : ""}`}
              style={{ ["--swatch" as string]: intToCss(p.color) }}
              onClick={() => {
                setTool("pen");
                setPenColor(p.color);
              }}
            />
          ))}
          <span className="tool-sep" />
          {PEN_WIDTHS.map((w) => (
            <button
              key={w.name}
              title={`ponta ${w.name}`}
              className={`nib ${penWidth === w.width ? "active" : ""}`}
              onClick={() => setPenWidth(w.width)}
            >
              <span style={{ width: w.width * 2.6, height: w.width * 2.6 }} />
            </button>
          ))}
          <span className="tool-sep" />
          <button
            className={`tool-btn ${tool === "eraser" ? "active" : ""}`}
            onClick={() => setTool(tool === "eraser" ? "pen" : "eraser")}
          >
            borracha
          </button>
        </div>

        <div className="tool-group">
          <button className="tool-btn" onClick={undo} disabled={undoRef.current.length === 0}>
            desfazer
          </button>
          <button
            className={`tool-btn ${confirmClear ? "danger" : ""}`}
            onClick={clearPage}
            onBlur={() => setConfirmClear(false)}
            disabled={!hasInk}
          >
            {confirmClear ? "apagar página?" : "limpar"}
          </button>
        </div>

        <div className="tool-group page-nav" aria-label="páginas">
          <button
            className="tool-btn"
            onClick={() => switchPage(pageInfo.cur - 1)}
            disabled={pageInfo.cur === 0}
            title="página anterior"
          >
            ◀
          </button>
          <span className="page-indicator">
            {pageInfo.cur + 1}/{pageInfo.total}
          </span>
          <button
            className="tool-btn"
            onClick={() => switchPage(pageInfo.cur + 1)}
            disabled={pageInfo.cur >= pageInfo.total - 1}
            title="próxima página"
          >
            ▶
          </button>
          <button className="tool-btn" onClick={newPage} title="nova página">
            + página
          </button>
        </div>

        <div
          className={`toolbar-status ${status.kind === "error" ? "error" : ""}`}
          role="status"
        >
          {status.kind === "busy" && <span className="status-dot" />}
          {status.msg ?? ""}
        </div>

        <div className="tool-group tool-group-right">
          <button
            className="tool-btn"
            onClick={() => void sketchGuess()}
            disabled={!hasInk || status.kind === "busy"}
            title="classificar o desenho mais recente"
          >
            desenho?
          </button>
          <label className={`auto-chip ${auto ? "on" : ""}`} title="detectar contas ao pausar a escrita">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
            auto
          </label>
          <button
            className="tool-btn solve-btn"
            onClick={() => void solve()}
            disabled={!hasInk || status.kind === "busy"}
          >
            {status.kind === "busy" ? "resolvendo…" : "resolver ="}
          </button>
        </div>
      </div>

      <div ref={wrapRef} className="notebook-wrap">
        <canvas
          ref={canvasRef}
          className={`notebook ${tool === "eraser" ? "erasing" : ""}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        />
        <div className={`notebook-hint ${hasInk ? "hidden" : ""}`}>
          anote livremente — termine uma conta com <span className="hint-eq">=</span> e ela se resolve
        </div>
      </div>
    </div>
  );
}
