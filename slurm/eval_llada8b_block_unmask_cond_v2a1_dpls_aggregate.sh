#!/bin/bash
# Aggregate the alpha=1 DPLS block_unmask eval (eval_llada8b_block_unmask_cond_v2a1_dpls_deltaai.sbatch)
# next to the GSM8K fixed-(b,tau) grid, BL32, and the Bernoulli v2 a1 run. CPU only.
set -euo pipefail
if [ "$(uname -m)" = aarch64 ]; then  # DeltaAI
  module load python/miniforge3_pytorch
  eval "$(conda shell.bash hook)"
else                                  # Delta
  source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
fi
conda activate dllm
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/llada8b_block_unmask_cond_v2_a1_dpls}
cd /u/zsun9/dllm-sampler-rl
# $RESULTS is globbed recursively, so it already covers $RESULTS/baselines/*.
python -m eval.aggregate_results --results_dir "$RESULTS" \
  --results_dir "$WORK/eval_results/blocksweep" \
  --results_dir "$WORK/eval_results/llada8b_bl32" \
  --results_dir "$WORK/eval_results/llada8b_block_unmask_cond_v2_a1" \
  --output_dir "$RESULTS"
python -m eval.analyze_block_policy "$RESULTS" --csv "$RESULTS/block_actions.csv"
