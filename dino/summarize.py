"""Compare the 4 runs (teacher temperature x {standard, fixed} CE) -> results/summary.{md,png}.

    uv run python -m dino.summarize --root outputs

Metrics (per run): final / best k-NN top-1, loss spikes, non-finite / skipped steps, late-training loss std,
final feature std and mean cosine (collapse), teacher entropy, gradient relative error vs float64 (median over
training; both implementations are measured on every run's own logits), temperature sensitivity.
Loss spike (fixed before the runs): finite step loss more than 6 robust SDs (1.4826 x MAD) above the median of the
previous 200 steps, and at least 0.05 above it.
Follow-up rule (fixed before the runs; one seed, so a signal only): at a given temperature, Fixed CE is "worth
following up" if k-NN top-1 is >= +1.0 %p over Standard CE, or loss spikes drop by >= 50 % with k-NN not worse by
more than 0.5 %p.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

RUNS = [("t0.04_standard", 0.04, "standard"), ("t0.04_fixed", 0.04, "fixed"),
        ("t0.1_standard", 0.10, "standard"), ("t0.1_fixed", 0.10, "fixed")]
SPIKE_WIN, SPIKE_Z, SPIKE_ABS = 200, 6.0, 0.05


def read(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def spikes(loss):
    idx = []
    for t in range(SPIKE_WIN, len(loss)):
        w = loss[t - SPIKE_WIN:t]
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med)) + 1e-8
        if loss[t] > med + max(SPIKE_Z * mad, SPIKE_ABS):
            idx.append(t)
    return idx


def run_stats(d):
    steps, knn, log, gerr = read(d / "steps.csv"), read(d / "knn.csv"), read(d / "log.csv"), read(d / "graderr.csv")
    if not steps:
        return None
    loss = np.array([float(s["loss"]) for s in steps if s["finite"] == "1"])
    sp = spikes(loss)
    last_ep = max(int(s["epoch"]) for s in steps)
    late = np.array([float(s["loss"]) for s in steps if s["finite"] == "1" and int(s["epoch"]) > 0.8 * last_ep])
    r = {"epochs_done": last_ep + 1, "done": (d / "DONE").exists(), "n_spikes": len(sp),
         "max_jump": float(max((loss[t] - np.median(loss[t - SPIKE_WIN:t]) for t in sp), default=0.0)),
         "n_nonfinite": sum(s["finite"] == "0" for s in steps), "n_skipped": sum(int(s["skipped"]) for s in steps),
         "late_loss_std": float(late.std()) if len(late) else float("nan"),
         "final_loss": float(loss[-50:].mean()) if len(loss) else float("nan"),
         "teacher_entropy": float(log[-1]["teacher_entropy"]) if log else float("nan"),
         "loss_curve": loss, "spike_idx": sp,
         "knn_curve": [(int(k["epoch"]) + 1, float(k["knn_top1"])) for k in knn]}
    if knn:
        r.update(knn_final=float(knn[-1]["knn_top1"]), knn_best=max(float(k["knn_top1"]) for k in knn),
                 feat_std=float(knn[-1]["feat_std"]), mean_cos=float(knn[-1]["mean_cos"]))
    for impl in ("standard", "fixed"):
        for dt in ("fp32", "bf16"):
            v = [float(g["rel_err_median"]) for g in gerr if g["impl"] == impl and g["dtype"] == dt]
            r[f"gerr_{impl}_{dt}"] = float(np.median(v)) if v else float("nan")
            r[f"gerr_curve_{impl}_{dt}"] = [(int(g["it"]), float(g["rel_err_median"])) for g in gerr
                                            if g["impl"] == impl and g["dtype"] == dt]
    return r


def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def e(x):
    return "–" if x is None or np.isnan(x) else f"{x:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("OUTPUT_ROOT", "outputs"))
    a = ap.parse_args()
    root = Path(a.root)
    res = Path(os.environ.get("RESULTS_DIR", "results"))
    res.mkdir(parents=True, exist_ok=True)
    S = {name: run_stats(root / name) for name, _, _ in RUNS}
    md = ["# DINO ViT-S/16 · CIFAR-100: Standard CE vs Fixed CE × teacher temperature", "",
          f"runs under `{root}`. One seed (42): differences are a preliminary signal, not significance.", "",
          "## 1. 결과표", "",
          "| run | τ_t | loss | epochs | k-NN top-1 (최종 / 최고) | loss spike | non-finite / skipped | 후반 loss std | "
          "feat std | mean cos | teacher entropy |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, tt, impl in RUNS:
        r = S[name]
        if r is None:
            md.append(f"| {name} | {tt} | {impl} | 0 | – | – | – | – | – | – | – |")
            continue
        md.append(f"| {name} | {tt} | {impl} | {r['epochs_done']}{'' if r['done'] else ' (진행 중)'} | "
                  f"{f(r.get('knn_final'))} / {f(r.get('knn_best'))} | {r['n_spikes']} | "
                  f"{r['n_nonfinite']} / {r['n_skipped']} | {f(r['late_loss_std'], 4)} | {f(r.get('feat_std'), 3)} | "
                  f"{f(r.get('mean_cos'), 3)} | {f(r['teacher_entropy'], 2)} |")
    md += ["", "## 2. Gradient 상대 오차 (float64 기준, 학습 중 측정값의 중앙값)", "",
           "각 run의 실제 logit으로 두 구현을 모두 측정 (student view 1 vs teacher view 0).", "",
           "| run | standard fp32 | fixed fp32 | standard bf16 | fixed bf16 |", "|---|---|---|---|---|"]
    for name, _, _ in RUNS:
        r = S[name]
        if r:
            md.append(f"| {name} | {e(r['gerr_standard_fp32'])} | {e(r['gerr_fixed_fp32'])} | "
                      f"{e(r['gerr_standard_bf16'])} | {e(r['gerr_fixed_bf16'])} |")
    md += ["", "## 3. Temperature 민감도 (τ_t 0.10 − 0.04, 최종 k-NN)", ""]
    for impl in ("standard", "fixed"):
        a4, a10 = S[f"t0.04_{impl}"], S[f"t0.1_{impl}"]
        if a4 and a10 and "knn_final" in a4 and "knn_final" in a10:
            md.append(f"- {impl}: {a10['knn_final'] - a4['knn_final']:+.2f}%p "
                      f"({a4['knn_final']:.2f} → {a10['knn_final']:.2f})")
    md += ["", "## 4. 판정 (사전 고정 규칙, seed 1개 → 예비 신호)", ""]
    for tt in ("0.04", "0.1"):
        s_, x_ = S[f"t{tt}_standard"], S[f"t{tt}_fixed"]
        if not (s_ and x_ and "knn_final" in s_ and "knn_final" in x_):
            md.append(f"- τ_t {tt}: 미완료")
            continue
        dk = x_["knn_final"] - s_["knn_final"]
        fewer = s_["n_spikes"] > 0 and x_["n_spikes"] <= 0.5 * s_["n_spikes"]
        go = dk >= 1.0 or (fewer and dk >= -0.5)
        md.append(f"- τ_t {tt}: Fixed − Standard k-NN {dk:+.2f}%p, spike {s_['n_spikes']} → {x_['n_spikes']}, "
                  f"grad 오차(fp32) {e(s_['gerr_standard_fp32'])} → {e(x_['gerr_fixed_fp32'])} → "
                  + ("**후속 실험 가치 있음 (seed·temperature 확장)**" if go else "차이 없음"))
    md += ["", "그림: `summary.png` (loss 곡선과 spike, k-NN 곡선, gradient 오차)", ""]
    (res / "summary.md").write_text("\n".join(md) + "\n")
    plot(S, res / "summary.png")
    print("\n".join(md))
    print(f"wrote {res / 'summary.md'}")


def plot(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    col = {"standard": "#2a78d6", "fixed": "#e8590c"}
    ls = {0.04: "-", 0.10: "--"}
    for name, tt, impl in RUNS:
        r = S[name]
        if not r:
            continue
        lab = f"τ_t {tt} {impl}"
        L = r["loss_curve"]
        k = max(1, len(L) // 500)
        ax[0].plot(np.arange(0, len(L), k), L[::k], ls[tt], color=col[impl], lw=0.8, label=lab)
        if r["spike_idx"]:
            ax[0].scatter(r["spike_idx"], L[r["spike_idx"]], color=col[impl], s=10, marker="x")
        if r["knn_curve"]:
            ax[1].plot(*zip(*r["knn_curve"]), ls[tt], color=col[impl], marker="o", label=lab)
        c = r["gerr_curve_" + impl + "_fp32"]
        if c:
            ax[2].plot(*zip(*c), ls[tt], color=col[impl], lw=0.8, label=lab)
    ax[0].set(title="training loss (x = spike)", xlabel="iteration")
    ax[1].set(title="k-NN top-1 (%)", xlabel="epoch")
    ax[2].set(title="grad rel. error vs fp64 (own impl, fp32)", xlabel="iteration", yscale="log")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
