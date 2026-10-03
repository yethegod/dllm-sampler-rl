#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bu-a1-dpls-kodcode-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# KodCode-only twin of train_llada8b_block_unmask_cond_v2a1_dpls_deltaai.sbatch:
# alpha=1, DPLS unmask head, window-conditioned block head, trained on
# KodCode-Light-RL-10K (prompts > 512 tokens dropped). See the config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml.
#
# ~1060 optimizer steps (vs 1870 on gsm8k_and_math), but each forward is longer
# (prompt up to 512 + 256 generated) and the reward runs unit tests, so expect
# roughly the same 18-24h as the math run. Resubmitting this same file auto-resumes
# from the last checkpoint-*50 and the same wandb run.
#
# WATCH: still_masked must be 0; train reward/accuracy (KodCode pass@1); NFE and
# block_size/frac_*; code_eval errors in the log.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls_kodcode

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a1-dpls-kodcode-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

# --per_device_train_batch_size 32: 4*32 = generation_batch_size 128 (see the
# math sibling for why this is mandatory on 4 GPUs).
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
