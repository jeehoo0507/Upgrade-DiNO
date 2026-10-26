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
la, lb = [float(x["loss"]) for x in a], [float(x["loss"]) for x in b]
fa, fb = json.load(open("$R/final.json")), json.load(open("$OUTPUT_ROOT/t0.04_fixed/final.json"))
if "$GPUS" == "cpu":      # CPU is deterministic: the resumed run must be bit-identical
    assert la == lb, (la, lb)
    assert fa == fb, (fa, fb)
else:                     # GPU kernels are not bit-deterministic run to run: same up to rounding noise
    assert la[:4] == lb[:4], "epoch 0 (before the stop) must match exactly"
    assert all(abs(x - y) <= 1e-3 * abs(y) for x, y in zip(la, lb)), (la, lb)
    assert all(abs(fa[k] - fb[k]) <= max(2.5, 0.05 * abs(fb[k])) for k in fb), (fa, fb)
assert all(x["ema"] == "1" and x["center"] == "1" for x in a)       # every toy step stepped (no fp16 on CPU)
g = list(csv.DictReader(open("$R/graderr.csv")))
assert all(x["e2e_err_median"] for x in g) and all(x["rel_err_median"] == x["e2e_err_median"] for x in g if x["dtype"] == "fp32")
print("  ok   stopped after epoch 0, resumed: losses and final k-NN match the uninterrupted run"
      + (" (bit-identical on CPU)" if "$GPUS" == "cpu" else " (within GPU rounding noise)"))
PY
echo "################ 5. one process per run (lock) ################"
L=$SM/lock_run; mkdir -p "$L"
uv run python -c "import fcntl, time; f=open('$L/RUNNING.lock','w'); fcntl.flock(f, fcntl.LOCK_EX); time.sleep(30)" &
HOLD=$!; sleep 2
if uv run python -m dino.train --loss fixed --teacher-temp 0.04 --epochs 1 --out "$L" $EXTRA > "$SM/lock.log" 2>&1; then
  echo "FAIL a second process trained a locked run"; kill $HOLD; exit 1
fi
grep -q "already training" "$SM/lock.log"; kill $HOLD 2>/dev/null; wait $HOLD 2>/dev/null || true
echo "  ok   a second process refuses to train a run that is already being trained"
echo "################ 6. bf16 loss option (1 epoch) ################"
uv run python -m dino.train --loss fixed --teacher-temp 0.04 --epochs 1 --out "$SM/bf16_run" $EXTRA --loss-dtype bf16 \
  > "$SM/bf16.log" 2>&1 && test -f "$SM/bf16_run/DONE"
echo "  ok   --loss-dtype bf16 trains and evaluates"
echo "################ 7. phase 2 queue (2 toy runs) + phase-2 summary ################"
NO_WAIT=1 EPOCHS=1 QUEUE="bf16_t0.04_fixed t0.06_standard" ./scripts/run_phase2.sh > "$SM/phase2.log" 2>&1
grep -q "phase 2 done (rc=0)" "$SM/phase2.log"
uv run python -c "import json; c=json.load(open('$OUTPUT_ROOT/bf16_t0.04_fixed/config.json')); assert c['loss_dtype']=='bf16' and c['loss']=='fixed' and c['teacher_temp']==0.04, c; c=json.load(open('$OUTPUT_ROOT/t0.06_standard/config.json')); assert c['loss_dtype']=='fp32' and c['teacher_temp']==0.06, c"
P2=$RESULTS_DIR/summary_phase2.md
for s in "## 1. 전체 결과" "## 2. Q1 붕괴 경계" "Q1 붕괴 경계" "Q2 bf16 loss (τ 0.04)" "| bf16_t0.04_fixed | bf16 |" "| t0.06_standard | fp32 |"; do
  grep -qF "$s" "$P2" || { echo "FAIL phase-2 summary lacks: $s"; exit 1; }
done
test -s "$RESULTS_DIR/summary_phase2.png"
echo "  ok   phase-2 run names -> (tau, CE, loss dtype), queue, summary_phase2.md/png"
echo; echo "SMOKE OK in $(( $(date +%s) - T0 )) s. Summary: $S"
