#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=40:00:00
#SBATCH --job-name=llada8b-bs-hp-dense-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Block-size-only policy on DeltaAI GH200 (aarch64), hidden-state input,
# token-level block sizes: the action is the next block size b in 1..128, read
# from the dLLM's last-layer hidden state; the Fast-dLLM threshold is pinned at
# 0.9 and alpha=1. Companion to train_llada8b_blocksize_only_deltaai.sbatch
# (confidence input, b in {8,16,32,48,64,96,128}); the config header lists the
# three fields that differ.
#
# WATCH: block_size/mean and block_size/bin_le* (power-of-two bins of b), the
# block entropy (uniform over 128 sizes is ln 128 = 4.852 nats) and
# num_blocks_mean. A point mass near b=64 means the policy learned the measured
# constant; mass piling into bin_le8 means many decisions per row, which grows the
# rollout buffer (~67 MiB per decision slot at batch 32).
#
# The bar at alpha=1.0 is fixed b=64, tau=0.9: 80.67% at 78.0 NFE, E[R] 0.5734.
# threshold/frac_0.9 must read 1.0 on every step.
#
# /work/hdd is shared with Delta; train.py auto-resumes from the latest
# checkpoint in OUTPUT_DIR if one exists.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_blocksize_only_hidden_proj_dense

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
  --config configs/experiment_configs/llada_8b_instruct_dit_blocksize_only_hidden_proj_dense.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
