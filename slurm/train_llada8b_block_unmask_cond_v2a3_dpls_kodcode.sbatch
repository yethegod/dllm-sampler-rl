#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bu-a3-dpls-kodcode
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Delta (8xA100) KodCode-only DPLS block_unmask run at alpha=3, the fast end of the KodCode
# alpha sweep (0.5 / 1 / 3). Only alpha changes vs the alpha=1 run; see the config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha3_dpls_kodcode.yaml.
# No smoke: identical to the alpha=0.5 / alpha=1 KodCode runs apart from the reward exponent.
#
# The config ships per_device_train_batch_size 16 and 8*16 = 128, so no override.
# Resubmitting this same file auto-resumes from the last checkpoint and the same wandb run.
#
# WATCH: still_masked must be 0; block_size/frac_128 and block_entropy (collapse onto 128);
# train pass rate; NFE; code_eval errors in the log.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a3_dpls_kodcode

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a3-dpls-kodcode-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha3_dpls_kodcode.yaml \
  --output_dir "$OUTPUT_DIR"
