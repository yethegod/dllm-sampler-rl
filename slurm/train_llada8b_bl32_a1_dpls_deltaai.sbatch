#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bl32-a1-dpls-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Fixed-block ablation control: the paper's dit_confidence policy at b=32 with the
# joint a1 run's sampler (DPLS), reward (alpha=1) and data. See the config header:
# configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha1_dpls.yaml.
# Delta twin: train_llada8b_bl32_a1_dpls.sbatch, same OUTPUT_DIR and wandb run.
#
# NO CROSS-CLUSTER LOCK: do not let this and the Delta twin run at the same time;
# when one starts, scancel the other. Resubmitting auto-resumes from the last checkpoint.
#
# WATCH: still_masked must be 0; num_steps_mean / tokens_per_step_mean for the speed
# alpha=1 buys; reward vs the joint a1 run (bu-a1-dpls-v1) at matched steps.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32_a1_dpls

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bl32-a1-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

# --per_device_train_batch_size 32: 4*32 = generation_batch_size 128 (mandatory on
# 4 GPUs, see train_llada8b_block_unmask_cond_v2a1_dpls_deltaai.sbatch).
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha1_dpls.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
