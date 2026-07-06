"use client";

import { useEffect, useRef, useState } from "react";
import katex from "katex";
import type { SolvedConta } from "@/lib/notebook";

/** Uma expressão renderizada com KaTeX (fallback: código cru). */
function Tex({ latex }: { latex: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(latex, ref.current, { throwOnError: false });
    } catch {
      ref.current.textContent = latex;
    }
  }, [latex]);
  return <span ref={ref} className="solved-tex" />;
}

/**
 * Painel flutuante com o histórico de contas resolvidas no caderno.
 * Colapsado vira só um selo com a contagem — a folha é a protagonista.
 */
export default function SolvedPanel({ solved }: { solved: SolvedConta[] }) {
  const [open, setOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  // conta nova → abre e rola para o fim
  const count = solved.length;
  useEffect(() => {
    if (count > 0) setOpen(true);
  }, [count]);
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [count, open]);

  if (count === 0) return null;

  return (
    <aside className={`solved-panel ${open ? "open" : ""}`}>
      <button className="solved-head" onClick={() => setOpen(!open)}>
        <span className="solved-title">contas resolvidas</span>
        <span className="solved-count">{count}</span>
        <span className="solved-caret">{open ? "▾" : "▴"}</span>
      </button>
      {open && (
        <div className="solved-list" ref={listRef}>
          {solved.map((c, i) => (
            <div className="solved-item" key={`${c.at}-${i}`}>
              <Tex latex={c.latex} />
              <span className="solved-eq">{c.eq ? "→" : "="}</span>
              <span className="solved-result">{c.result}</span>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
