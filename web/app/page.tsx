"use client";

import { useState } from "react";
import NotebookCanvas from "@/components/NotebookCanvas";
import SolvedPanel from "@/components/SolvedPanel";
import type { SolvedConta } from "@/lib/notebook";

/**
 * Rosetta — o caderno.
 *
 * Uma folha só, longa, para anotar livremente. Contas terminadas em "=" são
 * detectadas no meio das anotações e respondidas em tinta laranja, que se
 * desenha sozinha (POST /page/process). O canvas de expressão única virou isto.
 */
export default function Home() {
  const [solved, setSolved] = useState<SolvedConta[]>([]);

  return (
    <main className="page page-notebook">
      <header className="masthead reveal reveal-1">
        <h1 className="wordmark">
          Rosetta<span className="dot">.</span>
        </h1>
        <span className="masthead-meta">
          caderno · escreva a conta, termine com <span className="arrow">=</span>
        </span>
      </header>

      <div className="reveal reveal-2 notebook-area">
        <NotebookCanvas onSolvedChange={setSolved} />
      </div>

      <SolvedPanel solved={solved} />

      <footer className="colophon">
        <span>tinta online → bigru → transformer → latex → sympy</span>
        <span>projeto rosetta</span>
      </footer>
    </main>
  );
}
