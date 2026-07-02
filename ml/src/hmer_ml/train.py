"""Loop de treino do seq2seq (teacher forcing).

Respeita a restrição de 1 GPU de notebook (VRAM ~6-8 GB):
  - AMP (mixed precision)          -> train.amp
  - acumulação de gradiente        -> train.grad_accum
  - bucketing/padding por compr.   -> train.bucket_by_length
  - checkpoint + resume            -> checkpoint.resume / checkpoint.dir

Uso:
  python -m hmer_ml.train --config configs/overfit_crohme.yaml   # Fase 1
  python -m hmer_ml.train --config configs/mathwriting.yaml      # Fase 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.dataset import HMERDataset, LengthBucketSampler, make_collate_fn
from .model import build_model
from .tokenizer import LatexTokenizer
from .utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .utils.config import load_config


def _build_tokenizer(cfg) -> LatexTokenizer:
    """Carrega o vocab do disco; se não existir, constrói a partir dos labels do treino."""
    vocab_path = Path(cfg.tokenizer.vocab_path)
    if vocab_path.exists():
        return LatexTokenizer.load(vocab_path)

    from .data.inkml import iter_inkml

    print(f"[train] vocab inexistente em {vocab_path} — construindo a partir de {cfg.data.root}")
    labels = [ink.label for _, ink in iter_inkml(cfg.data.root) if ink.label]
    tok = LatexTokenizer().build_vocab(labels)
    tok.save(vocab_path)
    print(f"[train] vocab: {tok.vocab_size()} tokens salvos em {vocab_path}")
    return tok


def _lr_lambda(warmup: int, kind: str = "inv_sqrt"):
    """Fator de LR por passo. Warmup linear, depois:

      - inv_sqrt: decaimento 1/sqrt(step) (default; estável p/ treino real).
      - constant: LR fixa após o warmup (ideal p/ overfit/memorização).
    """
    def fn(step: int) -> float:
        step = max(step, 1)
        if step < warmup:
            return step / warmup
        return 1.0 if kind == "constant" else math.sqrt(warmup / step)
    return fn


def train(cfg) -> None:
    torch.manual_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.train.get("amp", False)) and device == "cuda"
    print(f"[train] device={device} amp={amp}")

    tok = _build_tokenizer(cfg)
    ds = HMERDataset(
        cfg.data.root,
        tok,
        max_points=cfg.data.get("max_points", 1024),
        max_tokens=cfg.data.get("max_tokens", 256),
        subset_size=cfg.data.get("subset_size"),
        do_normalize=cfg.data.get("normalize", True),
        resample_step=cfg.data.get("resample_step"),
        augment_cfg=cfg.data.get("augment_cfg") if cfg.data.get("augment") else None,
    )
    if len(ds) == 0:
        raise SystemExit(f"nenhuma amostra em {cfg.data.root} — gere dados ou ajuste o caminho")
    print(f"[train] {len(ds)} amostras")

    collate = make_collate_fn(tok.pad_id)
    bs = cfg.train.batch_size
    nw = cfg.train.get("num_workers", 0)
    # persistent_workers evita re-spawn dos workers a cada época (caro no Windows/spawn)
    dl_kw = {"num_workers": nw, "persistent_workers": nw > 0, "pin_memory": device == "cuda"}
    if cfg.train.get("bucket_by_length", False):
        print("[train] indexando comprimentos p/ bucketing (1x, pode demorar)...")
        lengths = [ds.sample_length(i) for i in range(len(ds))]
        sampler = LengthBucketSampler(lengths, bs, seed=cfg.get("seed", 42))
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, **dl_kw)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate, **dl_kw)

    model = build_model(
        cfg, vocab_size=tok.vocab_size(), pad_id=tok.pad_id,
        bos_id=tok.bos_id, eos_id=tok.eos_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parâmetros: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(cfg.train.get("warmup_steps", 1), cfg.train.get("scheduler", "inv_sqrt"))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir = Path(cfg.checkpoint.dir)
    step, start_epoch = 0, 0
    resume = cfg.checkpoint.get("resume") or latest_checkpoint(ckpt_dir)
    if resume and Path(resume).exists():
        step, start_epoch = load_checkpoint(resume, model=model, optimizer=opt,
                                             scheduler=sched, scaler=scaler,
                                             map_location=device)
        print(f"[train] retomando de {resume} (step={step}, epoch={start_epoch})")

    grad_accum = cfg.train.get("grad_accum", 1)
    grad_clip = cfg.train.get("grad_clip", 0.0)
    label_smoothing = cfg.train.get("label_smoothing", 0.0)
    save_every = cfg.checkpoint.get("save_every_steps", 500)
    epochs = cfg.train.epochs

    model.train()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(batch)  # [B, L-1, V]
                tgt_out = batch["tgt"][:, 1:]  # alinha com teacher forcing
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=tok.pad_id,
                    label_smoothing=label_smoothing,
                )
            scaler.scale(loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % save_every == 0:
                    save_checkpoint(ckpt_dir / f"step_{step}.ckpt", model=model, optimizer=opt,
                                    scheduler=sched, scaler=scaler, step=step, epoch=epoch)
            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(f"[train] epoch {epoch:4d}  loss {avg:.4f}  lr {sched.get_last_lr()[0]:.2e}")

    save_checkpoint(ckpt_dir / "last.ckpt", model=model, optimizer=opt, scheduler=sched,
                    scaler=scaler, step=step, epoch=epochs)
    print(f"[train] concluído. checkpoint final: {ckpt_dir / 'last.ckpt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino HMER online")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
