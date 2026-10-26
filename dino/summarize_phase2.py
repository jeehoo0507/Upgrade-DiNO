"""Phase 2 summary: every finished run under the output root (temperature sweep + bf16 loss), read from each
run's config.json -> results/summary_phase2.{md,png}.

    uv run python -m dino.summarize_phase2 --root outputs

Questions (rules fixed before the runs; one seed -> preliminary signal):
  Q1 collapse boundary  Does Fixed CE move the teacher temperature at which DINO collapses (no temperature warmup)?
                        collapsed = final teacher entropy >= 0.9 ln(out_dim) OR test-feature mean cosine >= 0.9.
                        boundary(impl) = smallest tau_t whose run collapsed. "Fixed moves the boundary" if
                        boundary(fixed) > boundary(standard), or at the same tau Fixed is not collapsed while Standard is.
  Q2 bf16 loss          With the loss in bf16 (where standard CE's gradient error is large), is Fixed CE better?
                        "yes" if at the same tau, bf16 Fixed - bf16 Standard k-NN >= +1.0 %p, or bf16 Standard
                        collapses / has >= 2x the loss spikes while bf16 Fixed does not.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from dino.summarize import e, f, run_stats

ENT_FRAC, COS_COLLAPSE = 0.9, 0.9


def discover(root):
    runs = []
    for cfg in sorted(Path(root).glob("*/config.json")):
        c = json.loads(cfg.read_text())
        r = run_stats(cfg.parent)
        if r is None:
            continue
        r.update(name=cfg.parent.name, tt=float(c["teacher_temp"]), impl=c["loss"],
                 dtype=c.get("loss_dtype", "fp32"), out_dim=int(c["out_dim"]))
        ent_max = math.log(r["out_dim"])
        r["collapsed"] = (r["teacher_entropy"] >= ENT_FRAC * ent_max) or (r.get("mean_cos", 0.0) >= COS_COLLAPSE)
        runs.append(r)
    return sorted(runs, key=lambda r: (r["dtype"], r["tt"], r["impl"]))


def pick(runs, tt, impl, dtype):
    for r in runs:
        if abs(r["tt"] - tt) < 1e-9 and r["impl"] == impl and r["dtype"] == dtype:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("OUTPUT_ROOT", "outputs"))
    a = ap.parse_args()
    res = Path(os.environ.get("RESULTS_DIR", "results"))
    res.mkdir(parents=True, exist_ok=True)
    runs = discover(a.root)
    md = ["# Phase 2: collapse boundary (teacher temperature sweep) and bf16 loss", "",
          f"runs under `{a.root}` (all runs with a config.json, incl. phase 1). One seed (42).", "",
          f"collapsed = teacher entropy ≥ {ENT_FRAC} × ln(out_dim) or test-feature mean cos ≥ {COS_COLLAPSE}.", "",
          "## 1. 전체 결과", "",
          "| run | loss dtype | τ_t | CE | epochs | k-NN 최종 / 최고 | 붕괴 | teacher entropy | mean cos | spike | skipped | "
          "grad err (own impl, loss dtype) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        ge = r[f"gerr_{r['impl']}_{'bf16' if r['dtype'] == 'bf16' else 'fp32'}"]
        md.append(f"| {r['name']} | {r['dtype']} | {r['tt']:g} | {r['impl']} | {r['epochs_done']}"
                  f"{'' if r['done'] else ' (진행 중)'} | {f(r.get('knn_final'))} / {f(r.get('knn_best'))} | "
                  f"{'**예**' if r['collapsed'] else '아니오'} | {f(r['teacher_entropy'])} | {f(r.get('mean_cos'), 3)} | "
                  f"{r['n_spikes']} | {r['n_skipped']} | {e(ge)} |")

    md += ["", "## 2. Q1 붕괴 경계 (fp32 loss, warmup 없음)", "", "| τ_t | Standard k-NN (붕괴) | Fixed k-NN (붕괴) |",
           "|---|---|---|"]
    tts = sorted({r["tt"] for r in runs if r["dtype"] == "fp32"})
    for tt in tts:
        s, x = pick(runs, tt, "standard", "fp32"), pick(runs, tt, "fixed", "fp32")
        cell = lambda r: "–" if r is None else f"{f(r.get('knn_final'))} ({'붕괴' if r['collapsed'] else '정상'})"
        md.append(f"| {tt:g} | {cell(s)} | {cell(x)} |")

    both = [tt for tt in tts if all((lambda r: r is not None and r["done"])(pick(runs, tt, i, "fp32"))
                                    for i in ("standard", "fixed"))]          # compare only where both finished

    def boundary(impl):
        col = [tt for tt in both if pick(runs, tt, impl, "fp32")["collapsed"]]
        return min(col) if col else None
    bs, bx = boundary("standard"), boundary("fixed")
    md += ["", "## 3. 판정 (사전 고정 규칙)", ""]
    moved = [tt for tt in tts if (lambda s, x: s and x and s["done"] and x["done"] and s["collapsed"] and not x["collapsed"])(
        pick(runs, tt, "standard", "fp32"), pick(runs, tt, "fixed", "fp32"))]
    worse = [tt for tt in tts if (lambda s, x: s and x and s["done"] and x["done"] and x["collapsed"] and not s["collapsed"])(
        pick(runs, tt, "standard", "fp32"), pick(runs, tt, "fixed", "fp32"))]
    md.append(f"- **Q1 붕괴 경계** (두 CE가 모두 끝난 τ {both}만 비교): Standard {bs if bs is not None else '붕괴 없음'}, "
              f"Fixed {bx if bx is not None else '붕괴 없음'} → "
              + ("**Fixed CE가 경계를 옮김**" if (moved or (bs is not None and (bx is None or bx > bs))) and not worse
                 else ("Fixed CE가 오히려 먼저 붕괴" if worse else "경계 변화 없음"))
              + (f" (Fixed만 정상인 τ: {moved})" if moved else ""))
    bf = sorted({r["tt"] for r in runs if r["dtype"] == "bf16"})
    for tt in bf:
        s, x = pick(runs, tt, "standard", "bf16"), pick(runs, tt, "fixed", "bf16")
        s32, x32 = pick(runs, tt, "standard", "fp32"), pick(runs, tt, "fixed", "fp32")
        if not (s and x and s["done"] and x["done"] and "knn_final" in s and "knn_final" in x):
            md.append(f"- **Q2 bf16 loss (τ {tt:g})**: 미완료")
            continue
        dk = x["knn_final"] - s["knn_final"]
        stab = (s["collapsed"] and not x["collapsed"]) or (s["n_spikes"] >= max(1, 2 * x["n_spikes"]) and not x["collapsed"])
        ref = (f"; fp32 기준 Standard {f(s32.get('knn_final'))}, Fixed {f(x32.get('knn_final'))}" if s32 and x32 else "")
        md.append(f"- **Q2 bf16 loss (τ {tt:g})**: Fixed − Standard k-NN {dk:+.2f}%p ({s['knn_final']:.2f} → "
                  f"{x['knn_final']:.2f}), 붕괴 {'예' if s['collapsed'] else '아니오'} → {'예' if x['collapsed'] else '아니오'}, "
                  f"spike {s['n_spikes']} → {x['n_spikes']}, grad err(bf16) {e(s['gerr_standard_bf16'])} → "
                  f"{e(x['gerr_fixed_bf16'])}{ref} → " + ("**bf16에서 Fixed CE 이점 있음**" if dk >= 1.0 or stab else "차이 없음"))
    md += ["", "그림: `summary_phase2.png` (τ별 k-NN, bf16 vs fp32 loss 곡선)", ""]
    (res / "summary_phase2.md").write_text("\n".join(md) + "\n")
    plot(runs, res / "summary_phase2.png")
    print("\n".join(md))
    print(f"wrote {res / 'summary_phase2.md'}")


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    col = {"standard": "#2a78d6", "fixed": "#e8590c"}
    for impl in ("standard", "fixed"):
        pts = sorted((r["tt"], r.get("knn_final", np.nan)) for r in runs if r["impl"] == impl and r["dtype"] == "fp32")
        if pts:
            ax[0].plot(*zip(*pts), "-o", color=col[impl], label=f"fp32 {impl}")
        b = sorted((r["tt"], r.get("knn_final", np.nan)) for r in runs if r["impl"] == impl and r["dtype"] == "bf16")
        if b:
            ax[0].plot(*zip(*b), "s", color=col[impl], mfc="none", ms=9, label=f"bf16 {impl}")
    ax[0].set(title="final k-NN top-1 vs teacher temperature", xlabel="τ_t", ylabel="%")
    for r in runs:
        if r["dtype"] == "bf16" or (abs(r["tt"] - 0.04) < 1e-9):
            L = r["loss_curve"]
            k = max(1, len(L) // 500)
            ax[1].plot(np.arange(0, len(L), k), L[::k], "--" if r["dtype"] == "bf16" else "-", color=col[r["impl"]],
                       lw=0.8, label=f"{r['dtype']} {r['impl']} τ{r['tt']:g}")
    ax[1].set(title="training loss: bf16 vs fp32 loss", xlabel="iteration")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
