#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=smoke-block-unmask-cond-v2-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# 20-step single-GPU smoke test for the v2 joint (block size, per-position
# unmasking) policy, to run before the 4xGH200 job. What it must show:
#   - the GRPO ratio starts at 1.0 on the first inner iteration (loss exactly 0):
#     with block_unmask_split_loss both the unmask and the block ratio are 1
#   - block_size/frac_* covers all 7 candidates; block_entropy near ln 7 (a bit
#     below 1.946 now that the boundary scores are non-zero at init)
#   - grad_norm/block_head > 0 from step 1, and block_head/bias_* /
#     block_head/boundary_w_norm drifting by step 20 (v1 sat at |bias| < 0.001
#     over the same 20 steps)
#   - mean_unmask_prob ~ 0.12 (sigmoid(-2)): the unmask head kept the paper's
#     smart_init and did not get entangled with the block head
#   - no threshold/* scalars at all
#   - num_steps_mean sane (well below 256) and memory fits on one GH200
#
# 1 GPU => accelerate takes the simple launcher: no rendezvous, no port 29500
# collision with co-scheduled jobs (that constraint only bites eval.pipeline).
# Batch math: generation_batch_size(8) = per_device_train_batch_size(8) x 1 GPU,
# and pdbs(8) % num_generations(8) == 0, matching the trainer's asserts.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
# Defaults to the v2 config; override to smoke another yaml on the same path:
#   sbatch --export=ALL,CONFIG=...,TAG=... slurm/smoke_block_unmask_cond_v2_deltaai.sbatch
CONFIG=${CONFIG:-configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2.yaml}
TAG=${TAG:-cond_v2}
OUTPUT_DIR=$WORK/checkpoints/smoke_block_unmask_${TAG}

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

# Fresh dir every time: train.py auto-resumes from get_last_checkpoint(),
# and a stale smoke checkpoint would silently skip the steps we want to see.
rm -rf "$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

cd "$REPO"

accelerate launch --num_processes 1 \
  -m train.train \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 8 \
  --generation_batch_size 8 \
  --max_steps 20 \
  --save_steps 500 \
  --warmup_steps 5
