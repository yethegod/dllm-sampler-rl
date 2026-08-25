#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --job-name=probe-stage0
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Stage 0 go/no-go: is the observation the bottleneck at all?
#
# Every policy here sees one number per position (top-1 confidence). The hidden-state
# projector policies exist on the bet that the dLLM's 4096-d state separates good
# unmasking decisions better. This job tests that bet for ~1 GPU-hour instead of the
# multi-day run it would otherwise take to find out.
#
# Decoding is Fast-dLLM at thres 0.9 -- a FIXED baseline, never a policy under training,
# so the label cannot depend on the thing being evaluated. The label is free: x0 is
# already computed every step, and comparing it to the final sequence gives, for each
# still-masked position, "would unmasking here have committed the token that eventually
# won?".
#
# What the three AUCs mean:
#   hidden ~= top-1    the observation is not the bottleneck. Abandon the hidden-input
#                      line and look at reward / credit assignment instead -- which is
#                      also what the block-policy adaptivity diagnosis pointed at.
#   top-16 ~= hidden   set confidences_top_p: 16 (one line, +1920 params) and skip the
#                      projector, the replay path, and the retuning that comes with them.
#   hidden >  both     the bet is live, and a dense-label warm-start has signal to teach.
#
# Sizing, corrected after the first run (jobs 21449052 / 21449753). That run collected
# 438k rows from only 88 examples and the 4096-d probe overfit at the example level:
# rows inside one generation share a trajectory, so the effective sample size is the
# number of EXAMPLES, not rows. This one trades rows for examples -- 384 examples,
# replaying every 4th step -- which costs about the same GPU time because the replay
# pass shrinks by the same factor. Fast-dLLM at 0.9 takes T=205-255 steps at
# block_length 32 (not the 81 recorded at block_length 128), so a batch is roughly
# T + T/4 forwards.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUT=$WORK/probe_data/${TAG:-gsm8k_t09_wide}.npz

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$WORK/probe_data" "$REPO/slurm/logs"
cd "$REPO"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== commit ==="
git -C "$REPO" log -1 --format='%h %s'

echo
if [ -n "${SKIP_COLLECT:-}" ] && [ -f "$OUT" ]; then
  echo "############ collect: skipped, reusing $OUT ############"
else
echo "############ collect ############"
python -u -m eval.collect_probe_data \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --dataset "${DATASET:-gsm8k}" \
  --n_test "${N_TEST:-384}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --thres "${THRES:-0.9}" \
  --step_stride "${STEP_STRIDE:-4}" \
  --rows_per_step "${ROWS_PER_STEP:-96}" \
  --out "$OUT"
fi

echo
echo "############ fit probes ############"
python -u -m eval.fit_probes --data "$OUT" --epochs "${EPOCHS:-400}"
