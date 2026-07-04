"""Loop de treino da cabeça de classificação de desenhos (Fase 4, QuickDraw).

Mesmo encoder de tinta da matemática (ADR 0006); só a perda muda: cross-entropy sobre
categorias, com acurácia de validação por época. Salva `last.ckpt` a cada época e
`best.ckpt` na melhor validação. Resume automático como no train.py.

Uso:  python -m hmer_ml.train_sketch --config ml/configs/quickdraw.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.dataset import LengthBucketSampler
from .data.quickdraw import QuickDrawDataset, SketchCollate
from .model import build_model
from .train import _lr_lambda
from .utils.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .utils.config import load_config


@torch.no_grad()
def evaluate_accuracy(model, loader, device: str) -> float:
    model.eval()
    hit = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        hit += int((logits.argmax(-1) == batch["label"]).sum())
        total += batch["label"].numel()
    model.train()
    return hit / max(total, 1)


def train(cfg) -> None:
    torch.manual_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg.train.get("amp", False)) and device == "cuda"
    print(f"[sketch] device={device} amp={amp}")

    d = cfg.data
    common = dict(
        root=d.root,
        categories=list(d.categories),
        train_per_class=d.get("train_per_class", 5000),
        val_per_class=d.get("val_per_class", 800),
        max_points=d.get("max_points", 512),
        resample_step=d.get("resample_step"),
    )
    train_ds = QuickDrawDataset(
        split="train",
        augment_cfg=d.get("augment_cfg") if d.get("augment") else None,
        **common,
    )
    val_ds = QuickDrawDataset(split="val", **common)
    print(f"[sketch] {len(train_ds)} treino / {len(val_ds)} val / "
          f"{len(d.categories)} classes")

    collate = SketchCollate()
    bs = cfg.train.batch_size
    nw = cfg.train.get("num_workers", 0)
    dl_kw = {"num_workers": nw, "persistent_workers": nw > 0, "pin_memory": device == "cuda"}
    if cfg.train.get("bucket_by_length", False):
        print("[sketch] indexando comprimentos p/ bucketing...")
        lengths = [train_ds.sample_length(i) for i in range(len(train_ds))]
        sampler = LengthBucketSampler(lengths, bs, seed=cfg.get("seed", 42))
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, **dl_kw)
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate, **dl_kw)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sketch] parâmetros: {n_params/1e6:.2f}M")

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
        print(f"[sketch] retomando de {resume} (step={step}, epoch={start_epoch})")

    grad_clip = cfg.train.get("grad_clip", 0.0)
    save_every = cfg.checkpoint.get("save_every_steps", 2000)
    best_acc = 0.0

    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running, hit, seen = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(batch)  # [B, C]
                loss = F.cross_entropy(logits, batch["label"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running += loss.item()
            hit += int((logits.argmax(-1) == batch["label"]).sum())
            seen += batch["label"].numel()
            if step % save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{step}.ckpt", model=model, optimizer=opt,
                                scheduler=sched, scaler=scaler, step=step, epoch=epoch)

        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"[sketch] epoch {epoch}  loss {running/max(seen//cfg.train.batch_size,1):.4f}  "
              f"train_acc {hit/max(seen,1):.3f}  val_acc {val_acc:.3f}  "
              f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        save_checkpoint(ckpt_dir / "last.ckpt", model=model, optimizer=opt, scheduler=sched,
                        scaler=scaler, step=step, epoch=epoch + 1)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(ckpt_dir / "best.ckpt", model=model, optimizer=opt, scheduler=sched,
                            scaler=scaler, step=step, epoch=epoch + 1)

    print(f"[sketch] concluído. melhor val_acc={best_acc:.3f} — checkpoint: "
          f"{ckpt_dir / 'best.ckpt'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino do classificador de desenhos")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
