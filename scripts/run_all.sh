#!/usr/bin/env bash
# The 4 preliminary runs (teacher temperature 0.04 / 0.10 x Standard / Fixed CE), then the summary. Resumable.
#   export DATA_ROOT=$HOME/dino/data OUTPUT_ROOT=$HOME/dino/outputs GPUS=0
#   setsid nohup ./scripts/run_all.sh > $HOME/dino.log 2>&1 < /dev/null &
# Options (env): P=1 (runs at the same time on one GPU; keep 1 on a 24 GB A5000),
#   EPOCHS=50, EXTRA="..." (extra dino.train args, same for all 4 runs), BACKUP_DIR=... (copy after each run)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"; : "${DATA_ROOT:?set DATA_ROOT}"
[[ -n "${GPUS:-}" && "$GPUS" != cpu ]] && export CUDA_VISIBLE_DEVICES=${GPUS%%,*}
export EPOCHS=${EPOCHS:-50} EXTRA=${EXTRA:-}
P=${P:-1}
mkdir -p "$OUTPUT_ROOT"
if [[ "$EXTRA" != *"--dataset fake"* ]]; then
  echo "[$(date '+%F %T')] CIFAR-100 (downloaded once into $DATA_ROOT)"
  uv run python -c "from torchvision import datasets; import os; r=os.environ['DATA_ROOT']; [datasets.CIFAR100(r, train=t, download=True) for t in (True, False)]"
fi
one() {
  local tt=$1 impl=$2 name="t${1}_${2}"
  [[ -f $OUTPUT_ROOT/$name/DONE ]] && { echo "skip  $name"; return 0; }
  echo "[$(date '+%F %T')] start $name"
  uv run python -m dino.train --loss "$impl" --teacher-temp "$tt" --epochs "$EPOCHS" --out "$OUTPUT_ROOT/$name" \
    --data-root "$DATA_ROOT" $EXTRA > "$OUTPUT_ROOT/$name.log" 2>&1 || { echo "FAIL  $name (see $OUTPUT_ROOT/$name.log)"; return 1; }
  echo "[$(date '+%F %T')] done  $name  $(grep -o 'k-NN top1 [0-9.]*' "$OUTPUT_ROOT/$name.log" | tail -1)"
  if [[ -n "${BACKUP_DIR:-}" ]]; then mkdir -p "$BACKUP_DIR" && cp -a "$OUTPUT_ROOT/$name" "$OUTPUT_ROOT/$name.log" "$BACKUP_DIR/"; fi
}
export -f one
printf "0.04 standard\n0.04 fixed\n0.1 standard\n0.1 fixed\n" | xargs -P "$P" -L 1 bash -c 'one "$0" "$1"'
uv run python -m dino.summarize --root "$OUTPUT_ROOT"
[[ -n "${BACKUP_DIR:-}" ]] && cp -a "${RESULTS_DIR:-results}" "$BACKUP_DIR/" || true
echo "[$(date '+%F %T')] all done: ${RESULTS_DIR:-results}/summary.md, summary.png"
