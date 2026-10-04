#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bu-a0p5-dpls-kodcode-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# DeltaAI (4xGH200) twin of train_llada8b_block_unmask_cond_v2a0p5_dpls_kodcode.sbatch (Delta):
# alpha=0.5 KodCode-only DPLS run, same OUTPUT_DIR and wandb run, so either cluster can
# resume the other's checkpoints (train.train handles the 8->4 GPU change).
# Config header: configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0p5_dpls_kodcode.yaml.
#
# NO CROSS-CLUSTER LOCK: do not let this and the Delta job (22655512 at submission) run
# at the same time; when one starts, scancel the other.
#
# WATCH: still_masked must be 0; train reward/accuracy (KodCode pass@1); NFE and
# block_size/frac_*; code_eval errors in the log.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a0p5_dpls_kodcode

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a0p5-dpls-kodcode-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

# --per_device_train_batch_size 32: 4*32 = generation_batch_size 128 (see the
# math sibling for why this is mandatory on 4 GPUs).
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0p5_dpls_kodcode.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
