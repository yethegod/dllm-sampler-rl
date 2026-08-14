#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=40:00:00
#SBATCH --job-name=llada8b-blocksize-only-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Block-size-only policy on DeltaAI GH200 (aarch64): the action is just the next
# block size, drawn from {8,16,32,48,64,96,128}; the Fast-dLLM threshold is
# pinned at 0.9 rather than learned. Companion to the joint (b, tau) run in
# train_llada8b_blocksize.sbatch.
#
# WATCH FOR EARLY COLLAPSE (formulation.md 9). Within a few hundred steps, check
# the TensorBoard scalars block_size/frac_* and the policy entropy (uniform over
# 7 actions is ln 7 = 1.946 nats). If the histogram converges to a point mass on
# b=64 the policy has learned the marginal -- i.e. the constant we already
# measured -- and the run should be killed rather than left to finish.
#
# The bar at alpha=1.0 is fixed b=64: 80.67% accuracy at 78.0 NFE, E[R] = 0.5734.
# The AdaBlock heuristic (ada0.3) is 79.83% at 85.3 NFE, E[R] = 0.5446.
# threshold/frac_0.9 must read 1.0 on every step -- anything else means the
# one-element threshold_candidates did not take effect.
#
# /work/hdd is shared with Delta; train.py auto-resumes from the latest
# checkpoint in OUTPUT_DIR if one exists.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_blocksize_only

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

# --per_device_train_batch_size 32 is mandatory on 4 GPUs: the config ships
# generation_batch_size 128, and 4*32 = 128 is what keeps steps_per_generation
# at 1. Without it TRL silently computes 2 and the trainer's _prepare_inputs and
# advantage slicing desynchronize (train/trainer.py asserts this, so it fails
# fast rather than corrupting advantages). 32 % num_generations 8 == 0, so the
# per-device batch still splits into whole groups.
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_blocksize_only.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
