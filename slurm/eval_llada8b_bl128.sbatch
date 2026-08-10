#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-llada8b-bl128
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_bl128
RESULTS=$WORK/eval_results/llada8b_bl128
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL128_mixture.yaml

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# block_length (128) flows from the config; do not pass --block_length.

# 1) Trained policy: last checkpoint (=1870), 3 seeds
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints 1870 \
  --datasets gsm8k \
  --seeds 42,43,44 \
  --temperatures 1.0 \
  --sampling_mode bernoulli-argmax \
  --save_path "$RESULTS" \
  --no_aggregate

# 2) Heuristic baselines (deterministic decoding, single seed). K256/K128 are
# gen-length NFE budgets (gen_length=256), unchanged from BL32; fastdllm reads
# block_length=128 from the config (2 blocks of 128).
python -m eval.pipeline \
  "baseline-low_confidence-K256;baseline-low_confidence-K128;baseline-fastdllm-t0.9" \
  "$CONFIG" \
  --checkpoints self \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --save_path "$RESULTS" \
  --no_aggregate

# 3) Aggregate everything into CSV
python -m eval.aggregate_results --results_dir "$RESULTS"
