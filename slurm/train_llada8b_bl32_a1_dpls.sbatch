#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bl32-a1-dpls
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Delta (8xA100) twin of train_llada8b_bl32_a1_dpls_deltaai.sbatch: fixed-block
# ablation control (paper's dit_confidence policy, b=32, DPLS, alpha=1). Same
# OUTPUT_DIR and wandb run, so either cluster resumes the other's checkpoints.
# Config: configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha1_dpls.yaml.
#
# NO CROSS-CLUSTER LOCK: when one twin starts, scancel the other.
# The config ships per_device_train_batch_size 16 and 8*16 = 128, so no override.
#
# WATCH: still_masked must be 0; num_steps_mean / tokens_per_step_mean; reward vs
# the joint a1 run (bu-a1-dpls-v1) at matched steps.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32_a1_dpls

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bl32-a1-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha1_dpls.yaml \
  --output_dir "$OUTPUT_DIR"
