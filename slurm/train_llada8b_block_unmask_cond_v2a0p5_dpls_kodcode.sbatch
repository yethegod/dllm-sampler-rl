#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bu-a0p5-dpls-kodcode
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Delta (8xA100) alpha=0.5 twin of train_llada8b_block_unmask_cond_v2a1_dpls_kodcode_deltaai.sbatch.
# Only alpha changes; see the config header for why:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0p5_dpls_kodcode.yaml.
#
# The config ships per_device_train_batch_size 16 and 8*16 = 128, so no override.
# Resubmitting this same file auto-resumes from the last checkpoint and the same wandb run.
#
# WATCH: still_masked must be 0; train reward/accuracy (KodCode pass@1); NFE and
# block_size/frac_*; code_eval errors in the log.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a0p5_dpls_kodcode

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a0p5-dpls-kodcode-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0p5_dpls_kodcode.yaml \
  --output_dir "$OUTPUT_DIR"
