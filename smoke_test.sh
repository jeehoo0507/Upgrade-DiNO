#!/usr/bin/env bash
# Whole pipeline at toy scale: loss unit tests -> 4 runs (synthetic CIFAR, small crops / head, 2 epochs) -> resume
# check -> summary. CPU ok (~5-10 min). Writes only under smoke/.
#   ./smoke_test.sh            # GPUS=0 ./smoke_test.sh to use a GPU
set -euo pipefail
cd "$(dirname "$0")"
export GPUS=${GPUS:-cpu}
[[ "$GPUS" == cpu ]] && export CUDA_VISIBLE_DEVICES=
SM="$(realpath -m smoke)"; rm -rf "$SM"; mkdir -p "$SM"
export OUTPUT_ROOT="$SM/outputs" DATA_ROOT="$SM/data" RESULTS_DIR="$SM/results" BACKUP_DIR="$SM/backup"
T0=$(date +%s)
echo "################ 1. loss unit tests ################"
uv run python tests/test_losses.py
echo "################ 2. 4 toy runs + summary ################"
export EPOCHS=2 EXTRA="--dataset fake --limit 64 --knn-limit 48 --batch-size 16 --global-size 64 --local-size 32 \
--local-crops 2 --out-dim 1024 --warmup-epochs 1 --eval-every 1 --grad-err-every 2 --grad-err-rows 8 --num-workers 0"
./scripts/run_all.sh
echo "################ 3. checks ################"
for r in t0.04_standard t0.04_fixed t0.1_standard t0.1_fixed; do
  d=$OUTPUT_ROOT/$r
  for f in DONE config.json steps.csv graderr.csv log.csv knn.csv final.json checkpoint.pt teacher_backbone.pt; do
    test -e "$d/$f" || { echo "FAIL $r lacks $f"; exit 1; }
  done
  test -f "$BACKUP_DIR/$r/DONE" || { echo "FAIL backup $r"; exit 1; }
done
uv run python - <<PY
import csv, json
for r in ("t0.04_standard", "t0.04_fixed", "t0.1_standard", "t0.1_fixed"):
    d = "$OUTPUT_ROOT/" + r
    cfg = json.load(open(d + "/config.json"))
    assert cfg["loss"] == r.split("_")[1] and abs(cfg["teacher_temp"] - float(r[1:].split("_")[0])) < 1e-9, cfg
    steps = list(csv.DictReader(open(d + "/steps.csv")))
    assert len(steps) == 2 * 4 and [int(s["it"]) for s in steps] == list(range(8)), len(steps)
    g = list(csv.DictReader(open(d + "/graderr.csv")))
    assert {(x["impl"], x["dtype"]) for x in g} == {(i, t) for i in ("standard", "fixed") for t in ("fp32", "bf16")}
    assert len(list(csv.DictReader(open(d + "/knn.csv")))) == 2
# same seed -> identical first-step teacher entropy across losses at the same temperature (paired runs)
a = list(csv.DictReader(open("$OUTPUT_ROOT/t0.04_standard/steps.csv")))[0]
b = list(csv.DictReader(open("$OUTPUT_ROOT/t0.04_fixed/steps.csv")))[0]
assert a["teacher_entropy"] == b["teacher_entropy"], (a["teacher_entropy"], b["teacher_entropy"])
print("  ok   configs, 8 steps each, grad-error rows for both impls x dtypes, 2 k-NN evals, paired first step")
PY
S=$RESULTS_DIR/summary.md
for s in "## 1. 결과표" "## 2. Gradient" "## 3. Temperature" "## 4. 판정" "| t0.1_fixed | 0.1 | fixed | 2 |"; do
  grep -qF "$s" "$S" || { echo "FAIL summary lacks: $s"; exit 1; }
done
test -s "$RESULTS_DIR/summary.png"
echo "################ 4. exact resume ################"
R=$SM/resume_run
uv run python -m dino.train --loss fixed --teacher-temp 0.04 --epochs 2 --out "$R" $EXTRA --stop-after-epoch 0 > "$SM/resume1.log" 2>&1
test ! -f "$R/DONE"
uv run python -m dino.train --loss fixed --teacher-temp 0.04 --epochs 2 --out "$R" $EXTRA > "$SM/resume2.log" 2>&1
grep -q "resuming from .* at epoch 1" "$SM/resume2.log" && test -f "$R/DONE"
uv run python - <<PY
import csv, json
a = list(csv.DictReader(open("$R/steps.csv")))
b = list(csv.DictReader(open("$OUTPUT_ROOT/t0.04_fixed/steps.csv")))
assert [int(x["it"]) for x in a] == list(range(8)), "steps.csv duplicated or missing rows"
assert [x["loss"] for x in a] == [x["loss"] for x in b], ([x["loss"] for x in a], [x["loss"] for x in b])
assert json.load(open("$R/final.json")) == json.load(open("$OUTPUT_ROOT/t0.04_fixed/final.json"))
assert all(x["ema"] == "1" and x["center"] == "1" for x in a)       # every toy step stepped (no fp16 on CPU)
g = list(csv.DictReader(open("$R/graderr.csv")))
assert all(x["e2e_err_median"] for x in g) and all(x["rel_err_median"] == x["e2e_err_median"] for x in g if x["dtype"] == "fp32")
print("  ok   stopped after epoch 0, resumed: every loss and the final k-NN identical to the uninterrupted run")
PY
echo "################ 5. bf16 loss option (1 epoch) ################"
uv run python -m dino.train --loss fixed --teacher-temp 0.04 --epochs 1 --out "$SM/bf16_run" $EXTRA --loss-dtype bf16 \
  > "$SM/bf16.log" 2>&1 && test -f "$SM/bf16_run/DONE"
echo "  ok   --loss-dtype bf16 trains and evaluates"
echo; echo "SMOKE OK in $(( $(date +%s) - T0 )) s. Summary: $S"
