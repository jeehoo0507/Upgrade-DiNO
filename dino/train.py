"""DINO pre-training (single GPU) with a selectable cross-entropy implementation.

Follows facebookresearch/dino main_dino.py: multi-crop, EMA teacher (no drop path), centering + sharpening,
AdamW with cosine lr / weight-decay / momentum schedules, per-parameter gradient clipping, last layer frozen
for the first epoch, fp16 autocast for the networks. The loss itself is always computed in float32 (outputs are
cast before the loss), so "standard" vs "fixed" differ only in how the cross-entropy is evaluated.

    uv run python -m dino.train --loss fixed --teacher-temp 0.04 --out outputs/t0.04_fixed

Per run directory: config.json, steps.csv (every iteration), graderr.csv (every --grad-err-every iterations),
log.csv (every epoch), knn.csv (every --eval-every epochs), checkpoint.pt (resume), final.json, DONE.
Rerunning the same command resumes from checkpoint.pt.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dino.data import NUM_CLASSES, DataAugmentationDINO, eval_transform, get_dataset
from dino.knn import extract, feature_stats, knn_classifier
from dino.losses import LOSSES, DINOLoss, grad_rel_error
from dino.model import DINOHead, MultiCropWrapper, build_backbone

STEP_FIELDS = ["it", "epoch", "loss", "finite", "skipped", "ema", "center", "grad_norm", "lr", "wd", "momentum", "scale",
               "teacher_entropy", "teacher_maxprob", "t_feat_std"]
GERR_FIELDS = ["it", "epoch", "impl", "dtype", "rel_err_median", "rel_err_mean", "rel_err_max",
               "e2e_err_median", "sum_abs_median"]
LOG_FIELDS = ["epoch", "loss_mean", "loss_std", "loss_max", "n_nonfinite", "n_skipped", "teacher_entropy",
              "t_feat_std", "epoch_time_s", "max_mem_gb"]
KNN_FIELDS = ["epoch", "knn_top1", "knn_top5", "feat_std", "mean_cos"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", choices=LOSSES, required=True)
    ap.add_argument("--teacher-temp", type=float, required=True)
    ap.add_argument("--dataset", default="cifar100", choices=list(NUM_CLASSES))
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64, help="DINO per-GPU default")
    ap.add_argument("--lr", type=float, default=5e-4, help="base lr for batch 256 (scaled linearly)")
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--weight-decay-end", type=float, default=0.4)
    ap.add_argument("--momentum-teacher", type=float, default=0.996)
    ap.add_argument("--clip-grad", type=float, default=3.0)
    ap.add_argument("--freeze-last-layer", type=int, default=1)
    ap.add_argument("--out-dim", type=int, default=65536)
    ap.add_argument("--no-norm-last-layer", action="store_true")
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    ap.add_argument("--warmup-teacher-temp-epochs", type=int, default=0,
                    help="0 = constant teacher temperature (the variable of this study)")
    ap.add_argument("--center-momentum", type=float, default=0.9)
    ap.add_argument("--loss-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="precision of the DINO loss / cross-entropy. fp32 = DINO default (this study's main runs); "
                         "bf16 = follow-up where CE rounding error is large (standard CE ~470%% vs fixed ~0.7%% "
                         "relative gradient error near agreement in the unit test)")
    ap.add_argument("--global-size", type=int, default=224)
    ap.add_argument("--local-size", type=int, default=96)
    ap.add_argument("--local-crops", type=int, default=6)
    ap.add_argument("--global-scale", type=float, nargs=2, default=(0.4, 1.0))
    ap.add_argument("--local-scale", type=float, nargs=2, default=(0.05, 0.4))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-t", type=float, default=0.07)
    ap.add_argument("--grad-err-every", type=int, default=50)
    ap.add_argument("--grad-err-rows", type=int, default=32, help="samples per gradient-error measurement")
    ap.add_argument("--limit", type=int, default=None, help="[smoke] train images")
    ap.add_argument("--knn-limit", type=int, default=None, help="[smoke] images per k-NN split")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="[smoke] exit after saving this epoch's checkpoint (tests exact resume)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------- helpers ---
def cosine_schedule(base, final, epochs, niter, warmup_epochs=0, start_warmup=0.0):
    warm = np.linspace(start_warmup, base, warmup_epochs * niter) if warmup_epochs > 0 else np.array([])
    iters = np.arange(epochs * niter - len(warm))
    sched = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
    out = np.concatenate((warm, sched))
    assert len(out) == epochs * niter
    return out


def param_groups(model):
    reg, noreg = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (noreg if n.endswith(".bias") or p.ndim == 1 else reg).append(p)
    return [{"params": reg}, {"params": noreg, "weight_decay": 0.0}]


def clip_gradients(model, clip):
    """DINO utils.clip_gradients: per-parameter norm clipping. Returns the total pre-clip norm."""
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        n = p.grad.data.norm(2)
        sq += float(n) ** 2
        coef = clip / (n + 1e-6)
        if coef < 1:
            p.grad.data.mul_(coef)
    return math.sqrt(sq)


def append_csv(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def truncate_csv(path, max_epoch_excl):
    """On resume: drop rows written by an epoch that did not finish."""
    if not path.exists():
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else None
    if fields is None:
        return
    keep = [r for r in rows if int(r["epoch"]) < max_epoch_excl]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(keep)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def knn_eval(teacher, args, device, amp):
    tf = eval_transform(args.global_size)
    tr = get_dataset(args.dataset, args.data_root, True, tf, args.knn_limit)
    te = get_dataset(args.dataset, args.data_root, False, tf, args.knn_limit)
    mk = lambda ds: torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=args.num_workers,
                                                pin_memory=device.type == "cuda")
    teacher.eval()
    ftr, ltr = extract(teacher.backbone, mk(tr), device, amp)
    fte, lte = extract(teacher.backbone, mk(te), device, amp)
    top1, top5 = knn_classifier(ftr, ltr, fte, lte, args.knn_k, args.knn_t, NUM_CLASSES[args.dataset], device=device)
    return {"knn_top1": top1, "knn_top5": top5, **feature_stats(fte)}


# ------------------------------------------------------------------------------------------- main ---
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"{out} already finished (DONE)")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), cfg.get(k)) for k in set(old) | set(cfg) if k not in ("num_workers", "stop_after_epoch")
                and old.get(k) != cfg.get(k)}
        if diff:
            sys.exit(f"[error] {out} holds a run with different settings: {diff}")
    cfg_path.write_text(json.dumps(cfg, indent=2))

    seed_all(args.seed)
    aug = DataAugmentationDINO(tuple(args.global_scale), tuple(args.local_scale), args.local_crops,
                               args.global_size, args.local_size)
    train_ds = get_dataset(args.dataset, args.data_root, True, aug, args.limit)
    niter = len(train_ds) // args.batch_size
    assert niter > 0, "dataset smaller than one batch"

    # same seed -> identical initial weights in every run
    s_backbone = build_backbone(args.arch, args.drop_path)
    dim = s_backbone.num_features
    student = MultiCropWrapper(s_backbone, DINOHead(dim, args.out_dim, norm_last_layer=not args.no_norm_last_layer))
    teacher = MultiCropWrapper(build_backbone(args.arch, 0.0),       # DINO teacher: no drop path
                               DINOHead(dim, args.out_dim, norm_last_layer=not args.no_norm_last_layer))
    teacher.load_state_dict(student.state_dict())
    student, teacher = student.to(device), teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()                      # no dropout / BN in ViT-S, so this only documents intent
    ncrops = 2 + args.local_crops
    crit = DINOLoss(args.out_dim, ncrops, args.loss, args.teacher_temp, args.student_temp, args.center_momentum,
                    args.warmup_teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).to(device)
    opt = torch.optim.AdamW(param_groups(student))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lr_s = cosine_schedule(args.lr * args.batch_size / 256.0, args.min_lr, args.epochs, niter, args.warmup_epochs)
    wd_s = cosine_schedule(args.weight_decay, args.weight_decay_end, args.epochs, niter)
    mom_s = cosine_schedule(args.momentum_teacher, 1.0, args.epochs, niter)

    start_epoch = 0
    ck_path = out / "checkpoint.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        crit.load_state_dict(ck["crit"])
        start_epoch = ck["epoch"] + 1
        print(f"resuming from {ck_path} at epoch {start_epoch}")
    for name in ("steps.csv", "graderr.csv", "log.csv", "knn.csv"):
        truncate_csv(out / name, start_epoch)

    print(f"device {device} amp {amp} | loss {args.loss} teacher_temp {args.teacher_temp} | train {len(train_ds)} "
          f"x {ncrops} crops, {niter} it/epoch, {args.epochs} epochs", flush=True)
    for epoch in range(start_epoch, args.epochs):
        # Every RNG is re-seeded from (seed, epoch) at the epoch start, so a run resumed at an epoch boundary
        # continues exactly as the uninterrupted run (data order, worker augmentation, drop path, k-NN loaders).
        seed_all(args.seed * 1000 + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator().manual_seed(args.seed * 1000 + epoch)     # same data order in every run
        loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen,
                                             num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                             drop_last=True, persistent_workers=False)
        student.train()
        t0, steps, gerrs = time.time(), [], []
        for i, (images, _) in enumerate(loader):
            it = epoch * niter + i
            for j, g in enumerate(opt.param_groups):
                g["lr"] = lr_s[it]
                if j == 0:
                    g["weight_decay"] = wd_s[it]
            images = [im.to(device, non_blocking=True) for im in images]
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                t_feat, t_out = teacher(images[:2], return_backbone=True)
                s_out = student(images)
            ldt = torch.float32 if args.loss_dtype == "fp32" else torch.bfloat16
            s_out, t_out = s_out.to(ldt), t_out.to(ldt)          # loss precision: fp32 (default) or bf16
            if args.grad_err_every and it % args.grad_err_every == 0:
                with torch.no_grad():
                    zs, zt = crit.logits(s_out.detach().float(), t_out.detach().float(), epoch)
                    zs1, zt0 = zs[1][: args.grad_err_rows], zt[0][: args.grad_err_rows]   # student view 1 vs teacher view 0
                for impl in LOSSES:
                    for dt, dn in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                        e, e2e, sabs = grad_rel_error(zs1, zt0, impl, dt)
                        gerrs.append({"it": it, "epoch": epoch, "impl": impl, "dtype": dn,
                                      "rel_err_median": f"{e.median():.4e}", "rel_err_mean": f"{e.mean():.4e}",
                                      "rel_err_max": f"{e.max():.4e}", "e2e_err_median": f"{e2e.median():.4e}",
                                      "sum_abs_median": f"{sabs.median():.4e}"})
            loss = crit(s_out, t_out, epoch).float()
            finite = bool(torch.isfinite(loss))
            opt.zero_grad(set_to_none=True)
            gnorm, skipped = float("nan"), 0
            if finite:
                scale_before = scaler.get_scale() if amp else 1.0
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = clip_gradients(student, args.clip_grad) if args.clip_grad else float("nan")
                if epoch < args.freeze_last_layer:
                    for n, p in student.named_parameters():
                        if "last_layer" in n:
                            p.grad = None
                scaler.step(opt)
                scaler.update()
                skipped = int(amp and scaler.get_scale() < scale_before)    # fp16 overflow -> step skipped
            else:
                skipped = 1
            stepped = finite and not skipped
            with torch.no_grad():
                if stepped:            # EMA only after a real optimizer step (no teacher drift on failed steps)
                    m = mom_s[it]
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(m).add_((1 - m) * ps.detach().data)
                centered = crit.update_center(t_out)        # teacher is unchanged by a failed step: still valid
                tf = torch.nn.functional.normalize(t_feat.float(), dim=-1)
                t_std = float(tf.std(0).mean() * math.sqrt(tf.shape[1]))
            steps.append({"it": it, "epoch": epoch, "loss": f"{float(loss):.6f}", "finite": int(finite),
                          "skipped": skipped, "ema": int(stepped), "center": int(centered), "grad_norm": f"{gnorm:.4f}", "lr": f"{lr_s[it]:.3e}",
                          "wd": f"{wd_s[it]:.4f}", "momentum": f"{mom_s[it]:.6f}",
                          "scale": f"{scaler.get_scale() if amp else 1.0:.0f}",
                          "teacher_entropy": f"{crit.last['teacher_entropy']:.4f}",
                          "teacher_maxprob": f"{crit.last['teacher_maxprob']:.4f}", "t_feat_std": f"{t_std:.4f}"})
            if i % 50 == 0:
                print(f"ep {epoch} it {i}/{niter} loss {float(loss):.4f} gnorm {gnorm:.2f} lr {lr_s[it]:.2e} "
                      f"H(t) {crit.last['teacher_entropy']:.2f}", flush=True)
        dt_ep = time.time() - t0
        losses = np.array([float(s["loss"]) for s in steps if s["finite"]])
        append_csv(out / "steps.csv", STEP_FIELDS, steps)
        append_csv(out / "graderr.csv", GERR_FIELDS, gerrs)
        append_csv(out / "log.csv", LOG_FIELDS, [{
            "epoch": epoch, "loss_mean": f"{losses.mean():.5f}" if len(losses) else "nan",
            "loss_std": f"{losses.std():.5f}" if len(losses) else "nan",
            "loss_max": f"{losses.max():.5f}" if len(losses) else "nan",
            "n_nonfinite": sum(1 - s["finite"] for s in steps), "n_skipped": sum(s["skipped"] for s in steps),
            "teacher_entropy": f"{np.mean([float(s['teacher_entropy']) for s in steps]):.4f}",
            "t_feat_std": f"{np.mean([float(s['t_feat_std']) for s in steps]):.4f}", "epoch_time_s": f"{dt_ep:.1f}",
            "max_mem_gb": f"{torch.cuda.max_memory_allocated() / 2**30:.2f}" if device.type == "cuda" else "0"}])
        last = epoch + 1 == args.epochs
        if last or (args.eval_every and (epoch + 1) % args.eval_every == 0):
            r = knn_eval(teacher, args, device, amp)
            append_csv(out / "knn.csv", KNN_FIELDS, [{"epoch": epoch, **{k: f"{v:.4f}" for k, v in r.items()}}])
            print(f"== epoch {epoch}: k-NN top1 {r['knn_top1']:.2f} top5 {r['knn_top5']:.2f} "
                  f"feat_std {r['feat_std']:.3f} mean_cos {r['mean_cos']:.3f}", flush=True)
        torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "crit": crit.state_dict(), "epoch": epoch},
                   str(ck_path) + ".tmp")
        os.replace(str(ck_path) + ".tmp", ck_path)
        mem = f", peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
        print(f"epoch {epoch} done in {dt_ep:.0f}s (ETA {dt_ep * (args.epochs - epoch - 1) / 3600:.1f} h){mem}",
              flush=True)
        if args.stop_after_epoch is not None and epoch == args.stop_after_epoch:
            print(f"stop after epoch {epoch} (--stop-after-epoch)")
            return
    if "r" not in locals():            # resumed after the last epoch had already been saved
        r = knn_eval(teacher, args, device, amp)
    torch.save({"teacher_backbone": teacher.backbone.state_dict(), "config": cfg}, out / "teacher_backbone.pt")
    (out / "final.json").write_text(json.dumps(r, indent=2))
    (out / "DONE").write_text("")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
