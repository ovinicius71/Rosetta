// Cliente da API. Passa pelo proxy /api/* (ver next.config.mjs).
import type { Ink } from "./ink";

export interface RecognizeResponse {
  latex: string;
}

export interface EvaluateResponse {
  result?: string;
  error?: string;
}

export async function recognize(ink: Ink): Promise<RecognizeResponse> {
  const res = await fetch("/api/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ink),
  });
  if (!res.ok) {
    // 501 enquanto o modelo não está treinado (stub da API) — tratar na UI.
    throw new Error(`recognize falhou: ${res.status}`);
  }
  return res.json();
}

export async function evaluate(latex: string): Promise<EvaluateResponse> {
  const res = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ latex }),
  });
  if (!res.ok) throw new Error(`evaluate falhou: ${res.status}`);
  return res.json();
}

// ── caderno: página inteira → contas resolvidas ─────────────────────────────
// Espelha PageInk/PageProcessResponse de api/schemas.py.

export interface PageStrokeDto {
  x: number[];
  y: number[];
  color?: number;
  width?: number;
}

export interface ExpressionResultDto {
  latex: string;
  result: string | null; // valor (contas) ou solução (equações 1-var)
  kind: string | null; // tipo da curva/superfície ("circunferencia", "esfera"…)
  description: string | null; // descrição pt-BR quando a equação foi classificada
  error: string | null;
  strokes: PageStrokeDto[]; // tinta do resultado, pronta para desenhar
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
  }
}

export async function processPage(
  strokes: PageStrokeDto[]
): Promise<{ expressions: ExpressionResultDto[] }> {
  const res = await fetch("/api/page/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strokes }),
  });
  if (!res.ok) throw new ApiError(res.status, `page/process falhou: ${res.status}`);
  return res.json();
}

export interface SketchGuessDto {
  label: string;
  label_pt: string;
  confidence: number;
}

export interface SketchResponseDto extends SketchGuessDto {
  topk: SketchGuessDto[];
  strokes: PageStrokeDto[]; // rótulo em tinta, pronto para desenhar
}

export async function recognizeSketch(strokes: PageStrokeDto[]): Promise<SketchResponseDto> {
  const res = await fetch("/api/sketch/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strokes }),
  });
  if (!res.ok) throw new ApiError(res.status, `sketch/recognize falhou: ${res.status}`);
  return res.json();
}
