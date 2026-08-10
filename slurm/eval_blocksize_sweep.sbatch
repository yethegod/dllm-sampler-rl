#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --array=0-12
#SBATCH --job-name=blocksweep
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Phase 0 for the learned adaptive-block-size project (see formulation.md §7,
# baseline 0): does block length actually buy anything on the accuracy/NFE
# plane? Already measured at thres=0.9: BL32 -> 79.45% @ 81.50 NFE,
# BL128 -> 71.11% @ 81.26 NFE. NFE is flat there, so this sweep adds lower
# thresholds where the block length can actually cap parallel unmasking.
#
# Eval is batch_size 1, so these are small jobs that backfill much faster than
# one 8-GPU job on a partition with ~2k pending.
#
# Two placement constraints, both learned the hard way:
#
# 1. Never 1 GPU. eval/eval.py:397 (CustomDistributedSampler) calls
#    dist.get_world_size() unconditionally, and eval.pipeline launches with
#    --num_processes $(cuda device count). At 1 process accelerate takes the
#    simple launcher, never initializes a process group, and eval dies before
#    the first example. >=2 processes forces the distributed launcher.
#
# 2. All 4 GPUs on the node, even though 2 would be enough throughput-wise.
#    eval.pipeline hardcodes its `accelerate launch` line with no
#    --main_process_port, so every task rendezvouses on the default port 29500.
#    Two array tasks landing on the same node => EADDRINUSE and instant death
#    for the second one. These nodes have exactly 4 GPUs, so claiming all 4
#    guarantees one accelerate job per node. (The alternative is plumbing a
#    --main_process_port flag through eval/pipeline.py, which would then allow
#    denser packing.)

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=$WORK/eval_results/blocksweep
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml

# (thres, block_length) grid. thres=0.9 x {32,128} is omitted -- already in
# eval_results/llada8b_bl32 and eval_results/llada8b_bl128.
GRID=(
  "0.9 8"   "0.9 16"  "0.9 64"
  "0.7 8"   "0.7 16"  "0.7 32"  "0.7 64"  "0.7 128"
  "0.5 8"   "0.5 16"  "0.5 32"  "0.5 64"  "0.5 128"
)
read -r TAU BL <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

# One save_path per (thres, block): run_eval's output dir name encodes neither
# the threshold nor the block length, so a shared save_path would make the arms
# overwrite each other. block_length is still recorded inside the generations
# JSON, which is what aggregate_results groups on.
OUT=$RESULTS/t${TAU}_bl${BL}
mkdir -p "$OUT" "$REPO/slurm/logs"
cd "$REPO"

echo "=== task $SLURM_ARRAY_TASK_ID: thres=$TAU block_length=$BL -> $OUT ==="

# thres is parsed out of the baseline checkpoint name by
# eval/eval.py:parse_baseline_checkpoint; block length is an explicit flag.
# gen_length (256) flows from the config. Fast-dLLM decoding is deterministic
# (temperature 0.0), hence a single seed.
python -m eval.pipeline "baseline-fastdllm-t${TAU}" "$CONFIG" \
  --checkpoints self \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --block_length "$BL" \
  --save_path "$OUT" \
  --no_aggregate

echo "=== done: thres=$TAU block_length=$BL ==="
