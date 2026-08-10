#!/bin/bash
# Aggregate the Phase 0 thres x block_length sweep. Cheap (pandas over ~15 JSON
# files), so this runs on the login node rather than through the queue -- there
# is no bhta-delta-cpu account to submit a CPU job under.
#
# Run once the eval_blocksize_sweep array has finished:
#   bash slurm/aggregate_blocksize_sweep.sh
#
# Includes the two arms measured earlier (thres=0.9 at BL32 / BL128), which live
# in the older per-run result dirs. aggregate_results groups on
# (run, dataset, block_length, checkpoint, temperature): run carries the
# threshold (baseline-fastdllm-t0.7) and block_length is read from inside the
# generations JSON, so every (thres, block) arm lands on its own row.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache

cd "$REPO"

python -m eval.aggregate_results \
  --results_dir "$WORK/eval_results/blocksweep" \
  --results_dir "$WORK/eval_results/llada8b_bl32" \
  --results_dir "$WORK/eval_results/llada8b_bl128" \
  --output_dir "$WORK/eval_results/blocksweep"

echo
echo "=== accuracy / NFE by (threshold, block_length) ==="
python - <<'PY'
import pandas as pd
df = pd.read_csv("/work/hdd/bhta/zsun9/eval_results/blocksweep/summary_statistics.csv")
df = df[df["run"].str.startswith("baseline-fastdllm-")]
df["thres"] = df["run"].str.replace("baseline-fastdllm-t", "", regex=False)
out = (
    df[["thres", "block_length", "accuracy_mean", "avg_steps_mean"]]
    .sort_values(["thres", "block_length"])
    .rename(columns={"accuracy_mean": "accuracy", "avg_steps_mean": "NFE"})
)
print(out.to_string(index=False))
PY
