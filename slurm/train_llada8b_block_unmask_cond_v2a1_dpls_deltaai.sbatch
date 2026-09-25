#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-bu-a1-dpls-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# v2 joint (block size, per-position unmasking) run at alpha=1 with the unmask
# head sampled by DPLS: one uninterrupted job for the whole epoch (no chunk
# chain, no RUNNING.lock). See the config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls.yaml.
#
# 48h is the ghx4 limit. The Bernoulli alpha=1 run managed ~1236 steps in 20h at
# ~82 NFE per rollout; DPLS rolls out at ~78 NFE at init (2-GPU smoke, job
# 3205870), so the 1870-step epoch should fit. If it does not, resubmitting this
# same file auto-resumes from the last checkpoint-*50 and the same wandb run.
#
# WATCH: still_masked must be 0; block_size/frac_* and block_entropy (ln 7 =
# 1.946 nats uniform) for a block-head collapse; tokens_per_step_mean and
# num_steps_mean for the speed the alpha=1 term buys.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a1-dpls-v1}
export WANDB_RESUME=allow
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
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
