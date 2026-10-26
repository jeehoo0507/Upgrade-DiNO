#!/usr/bin/env bash
# Stop every Upgrade-DINO process: run_all.sh / run_phase2.sh, their xargs worker loop, the per-run bash shells and dino.train.
# (Killing only run_all.sh leaves xargs alive, which then starts the next run.) Finished runs are untouched;
# an interrupted run resumes from its last checkpoint on the next launch.
# Patterns are anchored to the start of the command line, so shells / editors that merely mention these names
# are never matched.
PATS=("^xargs -P [0-9]+ -L 1 bash -c one" "^bash (\./)?scripts/run_(all|phase2)\.sh|^bash .*/scripts/run_(all|phase2)\.sh"
      "^bash -c one " "^uv run python -m dino\.(train|summarize)" "python[0-9.]* -m dino\.(train|summarize)")
for p in "${PATS[@]}"; do pkill -f -- "$p"; done      # xargs first, so no new run is started
for i in $(seq 1 20); do
  alive=0; for p in "${PATS[@]}"; do pgrep -f -- "$p" >/dev/null && alive=1; done
  [[ $alive == 0 ]] && break; sleep 1
done
for p in "${PATS[@]}"; do pkill -9 -f -- "$p" 2>/dev/null; done
sleep 1
left=""; for p in "${PATS[@]}"; do left+=$(pgrep -af -- "$p"); done
[[ -z $left ]] && echo "stopped: no Upgrade-DINO process left" || { echo "still running:"; echo "$left"; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
exit 0
