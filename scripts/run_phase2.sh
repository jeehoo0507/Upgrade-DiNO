#!/usr/bin/env bash
# Phase 2, queued behind phase 1: waits until scripts/run_all.sh has finished, then runs (in priority order)
#   1) bf16 loss at tau_t 0.04, Standard vs Fixed     -> does Fixed CE help where the CE rounding error is large?
#   2) fp32 loss at tau_t 0.06 / 0.05 / 0.07           -> does Fixed CE move DINO's collapse boundary?
# then results/summary_phase2.{md,png} (dino.summarize_phase2, includes the phase-1 runs). Resumable; one process
# per run (RUNNING.lock). Stop everything with ./scripts/stop.sh.
#   export DATA_ROOT=$HOME/dino/data OUTPUT_ROOT=$HOME/dino/outputs GPUS=0 BACKUP_DIR=/shared/users/<me>/dino
#   setsid nohup ./scripts/run_phase2.sh > $HOME/dino_phase2.log 2>&1 < /dev/null &
# Options (env): QUEUE="..." (run names, see below), P=1, EPOCHS=50, EXTRA="...", NO_WAIT=1 (do not wait for phase 1)
# Run names: [bf16_]t<tau>_<standard|fixed>, e.g. bf16_t0.04_fixed, t0.06_standard.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
[[ -n "${GPUS:-}" && "$GPUS" != cpu ]] && export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
export EPOCHS=${EPOCHS:-50} EXTRA=${EXTRA:-}
P=${P:-1}
QUEUE=${QUEUE:-"bf16_t0.04_standard bf16_t0.04_fixed t0.06_standard t0.06_fixed t0.05_standard t0.05_fixed t0.07_standard t0.07_fixed"}
if [[ "${NO_WAIT:-0}" != 1 ]]; then
  while pgrep -f "^bash (\./)?scripts/run_all\.sh|^bash .*/scripts/run_all\.sh" >/dev/null; do
    echo "[$(date '+%F %T')] waiting for phase 1 (scripts/run_all.sh) to finish"; sleep 300
  done
fi
echo "[$(date '+%F %T')] phase 2: [$QUEUE], $P at a time, $EPOCHS epochs"
one() {
  local name=$1 dt=fp32 rest tt impl
  rest=$name
  [[ $rest == bf16_* ]] && { dt=bf16; rest=${rest#bf16_}; }
  tt=${rest%%_*}; tt=${tt#t}; impl=${rest#*_}
  [[ $impl == standard || $impl == fixed ]] || { echo "bad run name $name"; return 1; }
  [[ -f $OUTPUT_ROOT/$name/DONE ]] && { echo "skip  $name"; return 0; }
  echo "[$(date '+%F %T')] start $name (tau $tt, $impl CE, $dt loss)"
  uv run python -m dino.train --loss "$impl" --teacher-temp "$tt" --loss-dtype "$dt" --epochs "$EPOCHS" \
    --out "$OUTPUT_ROOT/$name" --data-root "$DATA_ROOT" $EXTRA > "$OUTPUT_ROOT/$name.log" 2>&1 \
    || { echo "FAIL  $name (see $OUTPUT_ROOT/$name.log)"; return 1; }
  echo "[$(date '+%F %T')] done  $name  $(grep -o 'k-NN top1 [0-9.]*' "$OUTPUT_ROOT/$name.log" | tail -1)"
  if [[ -n "${BACKUP_DIR:-}" ]]; then mkdir -p "$BACKUP_DIR" && cp -a "$OUTPUT_ROOT/$name" "$OUTPUT_ROOT/$name.log" "$BACKUP_DIR/"; fi
  uv run python -m dino.summarize_phase2 --root "$OUTPUT_ROOT" > /dev/null 2>&1 || true   # refresh after every run
}
export -f one
rc=0
for n in $QUEUE; do echo "$n"; done | xargs -P "$P" -L 1 bash -c 'one "$0"' || rc=$?
uv run python -m dino.summarize_phase2 --root "$OUTPUT_ROOT"
[[ -n "${BACKUP_DIR:-}" ]] && cp -a "${RESULTS_DIR:-results}" "$BACKUP_DIR/" || true
echo "[$(date '+%F %T')] phase 2 done (rc=$rc): ${RESULTS_DIR:-results}/summary_phase2.md, summary_phase2.png"
exit $rc
