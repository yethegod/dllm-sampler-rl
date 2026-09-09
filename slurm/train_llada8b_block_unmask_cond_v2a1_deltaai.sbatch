#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=20:00:00
#SBATCH --job-name=llada8b-block-unmask-cond-v2a1-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# v2 joint (block size, per-position unmasking) run at alpha=1 with the reward
# scored on productive steps (no stall charge). Diagnostic for job 3117175's
# collapse onto b=128: at alpha=1 the objective prefers b~64, and without the
# stall charge the training NFE is the eval NFE. See the config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1.yaml.
#
# WHAT TO WATCH beyond v1's collapse signals: block_head/bias_* and
# block_head/boundary_w_norm must move within the first ~100 steps and
# grad_norm/block_head must be non-zero from step 1 -- if either sits at its
# v1 value the fix did not take. block_size/frac_* leaving uniform is the
# point of the run; a point mass on one size within a few hundred steps is the
# collapse to watch for. 20h is ~1100 of 1870 steps (BL128 took 33h41m for an
# epoch); resubmit this same file to auto-resume from the last checkpoint-*50.
#
# WATCH FOR EARLY COLLAPSE (formulation.md 9). Within a few hundred steps, check
# the TensorBoard scalars block_size/frac_*, block_entropy (uniform over 7 is
# ln 7 = 1.946 nats) and entropy / mean_unmask_prob for the unmask head. A
# point mass on one block size, or mean_unmask_prob racing to 1.0 (unmask
# everything at once) or 0 (never finish), is a reason to kill the run rather
# than let it finish.
#
# /work/hdd is shared with Delta; train.py auto-resumes from the latest
# checkpoint in OUTPUT_DIR if one exists.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1

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
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
