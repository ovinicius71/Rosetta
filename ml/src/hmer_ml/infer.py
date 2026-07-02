"""Inferência: tinta (Ink / dict do esquema compartilhado) → LaTeX. Usado pela API.

Mantém a MESMA representação de tinta do treino (parse → normalize → features), garantindo
consistência treino/inferência (ADR 0004). Decodificação greedy (beam_size=1) ou beam
search com normalização por comprimento (beam_size>1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data.ink import Ink
from .model import build_model
from .tokenizer import LatexTokenizer
from .utils.checkpoint import load_checkpoint
from .utils.config import load_config


class Recognizer:
    """Carrega tokenizer + modelo uma vez e reconhece tinta sob demanda.

    A API instancia isto no startup (carregando um checkpoint) e chama `.recognize`.
    """

    def __init__(self, ckpt_path: str, config_path: str, device: str = "cpu"):
        self.device = device
        self.cfg = load_config(config_path)
        self.tok = LatexTokenizer.load(self.cfg.tokenizer.vocab_path)
        self.model = build_model(
            self.cfg,
            vocab_size=self.tok.vocab_size(),
            pad_id=self.tok.pad_id,
            bos_id=self.tok.bos_id,
            eos_id=self.tok.eos_id,
        ).to(device)
        load_checkpoint(ckpt_path, model=self.model, map_location=device)
        self.model.eval()

    def _ink_to_batch(self, ink: Ink) -> dict:
        """Ink → batch de 1 exemplo (sem label). Mesmo pré-processamento do treino."""
        from .data.ink import ink_to_features, normalize, resample

        if self.cfg.data.get("normalize", True):
            ink = normalize(ink)
        step = self.cfg.data.get("resample_step")
        if step:
            ink = resample(ink, step)
        feats = ink_to_features(ink)[: self.cfg.data.get("max_points", 1024)]
        src = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return {
            "src": src,
            "src_lengths": torch.tensor([src.size(1)], dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def recognize(self, ink: Ink | dict, beam_size: int = 4, max_len: int = 256) -> str:
        """Retorna o LaTeX reconhecido (greedy se beam_size<=1, senão beam search)."""
        if isinstance(ink, dict):
            ink = Ink.from_dict(ink)
        batch = self._ink_to_batch(ink)
        if beam_size <= 1:
            ids = self.model.greedy_decode(batch, max_len=max_len)[0]
        else:
            memory, mmask = self.model.encode(batch)
            ids = beam_search(
                self.model.head, memory, mmask,
                bos_id=self.tok.bos_id, eos_id=self.tok.eos_id,
                beam_size=beam_size, max_len=max_len,
            )
        return self.tok.decode(ids)


@torch.no_grad()
def beam_search(head, memory, memory_mask, *, bos_id: int, eos_id: int,
                beam_size: int, max_len: int, length_alpha: float = 0.7) -> list[int]:
    """Beam search autorregressivo para 1 exemplo (memory [1, T, D]).

    Score = soma de log-probs, normalizada por len**length_alpha na seleção final
    (GNMT-style), para não privilegiar hipóteses curtas. Retorna ids sem <bos>/<eos>.
    """
    device = memory.device
    # replica a memory para as k hipóteses (mesma tinta para todos os beams)
    mem_k = memory.expand(beam_size, -1, -1)
    mask_k = memory_mask.expand(beam_size, -1)

    ys = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((beam_size,), float("-inf"), device=device)
    scores[0] = 0.0  # só o beam 0 é válido no 1º passo (os demais são cópias do bos)
    done = torch.zeros(beam_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = head(mem_k, mask_k, ys)[:, -1]           # [k, vocab]
        logp = F.log_softmax(logits.float(), dim=-1)      # [k, vocab]
        # beams terminados só podem "continuar" com eos a custo zero
        logp[done] = float("-inf")
        logp[done, eos_id] = 0.0

        cand = scores.unsqueeze(1) + logp                 # [k, vocab]
        flat = cand.view(-1)
        top_scores, top_idx = flat.topk(beam_size)
        beam_idx = top_idx // logp.size(1)
        tok_idx = top_idx % logp.size(1)

        ys = torch.cat([ys[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        scores = top_scores
        done = done[beam_idx] | (tok_idx == eos_id)
        if bool(done.all()):
            break

    # seleção final com normalização por comprimento (sem contar bos/eos)
    def seq_ids(row: list[int]) -> list[int]:
        out = []
        for t in row[1:]:  # pula <bos>
            if t == eos_id:
                break
            out.append(t)
        return out

    best, best_score = [], float("-inf")
    for i in range(beam_size):
        ids = seq_ids(ys[i].tolist())
        norm = max(len(ids), 1) ** length_alpha
        s = scores[i].item() / norm
        if s > best_score:
            best, best_score = ids, s
    return best
