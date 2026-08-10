#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --job-name=smoke-blocksize
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# 20-step single-GPU smoke test for the joint (block size, threshold) policy.
# All prior verification used a mock dLLM on the login node; this is the first
# run against the real 8B model. What it must show:
#   - the GRPO ratio starts at 1.0 on the first inner iteration
#   - block_size/frac_* and threshold/frac_* land in the logs, near-uniform
#   - num_steps_mean is sane (well below 256) and memory fits on one A100
#
# 1 GPU => accelerate takes the simple launcher: no rendezvous, no port 29500
# collision with co-scheduled jobs (that constraint only bites eval.pipeline).
# Batch math: generation_batch_size(8) = per_device_train_batch_size(8) x 1 GPU,
# and pdbs(8) % num_generations(8) == 0, matching the trainer's asserts.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/smoke_blocksize

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

# Fresh dir every time: train.py auto-resumes from get_last_checkpoint(),
# and a stale smoke checkpoint would silently skip the steps we want to see.
rm -rf "$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

cd "$REPO"

accelerate launch --num_processes 1 \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 8 \
  --num_generations 8 \
  --generation_batch_size 8 \
  --max_steps 20 \
  --save_steps 500 \
  --warmup_steps 5
