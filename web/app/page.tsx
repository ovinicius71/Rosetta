"use client";

import { useState } from "react";
import InkCanvas from "@/components/InkCanvas";
import LatexView from "@/components/LatexView";
import { recognize } from "@/lib/api";
import type { Ink } from "@/lib/ink";

export default function Home() {
  const [latex, setLatex] = useState<string>("");
  const [status, setStatus] = useState<string>("");

  async function handleRecognize(ink: Ink) {
    setStatus("reconhecendo…");
    try {
      const { latex } = await recognize(ink);
      setLatex(latex);
      setStatus("");
    } catch (e) {
      // Enquanto o modelo não está treinado, a API devolve 501 (stub). Ver roadmap Fase 3.
      setStatus(`sem modelo ainda (${(e as Error).message})`);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: 24 }}>
      <h1>Rosetta — escreva matemática à mão</h1>
      <InkCanvas width={640} height={360} onRecognize={handleRecognize} />
      {status && <p style={{ opacity: 0.7 }}>{status}</p>}
      <LatexView latex={latex} />
    </main>
  );
}
