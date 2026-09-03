#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=smoke-block-unmask-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# 20-step single-GPU smoke test for the joint (block size, per-position
# unmasking) policy, to run before the 40h job. What it must show:
#   - the GRPO ratio starts at 1.0 on the first inner iteration (loss exactly 0):
#     the rollout's old log-probs and the trainer's recomputation use the same
#     [bernoulli | categorical] decomposition
#   - block_size/frac_* covers all 7 candidates, uniform at step 0 up to the
#     feasibility tilt toward small b late in a rollout; block_entropy ~ ln 7
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
OUTPUT_DIR=$WORK/checkpoints/smoke_block_unmask

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
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint.yaml \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 8 \
  --generation_batch_size 8 \
  --max_steps 20 \
  --save_steps 500 \
  --warmup_steps 5
