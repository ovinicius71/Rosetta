// Modelo do caderno — traços de página inteira (formato de /page/process, não o
// ink.schema.json): arrays paralelos x[]/y[] + estilo por traço. Espelha api/schemas.py
// (PageStroke). A tinta é a fonte da verdade; o canvas é só desenho.

export interface NoteStroke {
  x: number[];
  y: number[];
  color: number; // RGB inteiro (0xRRGGBB), como o backend e o Xournal++ usam
  width: number;
  result?: boolean; // tinta desenhada pelo sistema (resposta de uma conta)
}

/** Conta resolvida (para o painel lateral). */
export interface SolvedConta {
  latex: string;
  result: string;
  at: number; // epoch ms
}

// Cor da tinta de resultado — TEM de casar com hmer_ml.segment.RESULT_COLOR:
// é ela que marca uma conta como "já resolvida" nas chamadas seguintes.
export const RESULT_COLOR = 0xe8590c;

/** Canetas disponíveis (tinta azul-preta, grafite quente, siena). */
export const PENS = [
  { name: "tinta", color: 0x2a3670 },
  { name: "grafite", color: 0x4a4438 },
  { name: "siena", color: 0x8a4b28 },
] as const;

export const PEN_WIDTHS = [
  { name: "fina", width: 2 },
  { name: "média", width: 3.25 },
] as const;

export function intToCss(color: number): string {
  return `#${color.toString(16).padStart(6, "0")}`;
}

// ── persistência local (o caderno sobrevive ao reload) ──────────────────────
// v1 guardava uma página única; v2 guarda várias. A migração é automática no load.

const STORAGE_KEY = "rosetta-notebook-v1";

export const PAGE_MIN_HEIGHT = 1400;

export interface NotebookPage {
  id: number;
  strokes: NoteStroke[];
  pageHeight: number;
  solved: SolvedConta[];
}

export interface NotebookState {
  version: 2;
  pages: NotebookPage[];
  current: number;
}

export function emptyPage(): NotebookPage {
  return { id: Date.now(), strokes: [], pageHeight: PAGE_MIN_HEIGHT, solved: [] };
}

export function loadNotebook(): NotebookState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (s?.version === 2 && Array.isArray(s.pages) && s.pages.length > 0) {
      return s as NotebookState;
    }
    if (Array.isArray(s?.strokes)) {
      // v1 → v2: a página antiga vira a página 1
      return {
        version: 2,
        current: 0,
        pages: [
          {
            id: Date.now(),
            strokes: s.strokes,
            pageHeight: s.pageHeight || PAGE_MIN_HEIGHT,
            solved: s.solved ?? [],
          },
        ],
      };
    }
    return null;
  } catch {
    return null;
  }
}

export function saveNotebook(state: NotebookState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* quota/modo privado — o caderno só não persiste */
  }
}

export function clearNotebook(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* idem */
  }
}
