#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=01:00:00
#SBATCH --job-name=smoke-ddp
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# The chunk-wise backward has only ever run on one GPU, which cannot exercise the part
# most likely to be wrong. Two things here are invisible at --num_processes 1:
#
#   1. no_sync scope. DDP decides whether to prepare for reduction inside forward(),
#      so a context wrapping only backward() leaves every chunk syncing. Worse, the
#      number of chunks depends on T, which varies per rank with the rollout length,
#      so a per-chunk collective mismatches across ranks and hangs. Fixed by syncing
#      only on the final chunk, whose backward Trainer.training_step performs -- one
#      reducing backward per rank regardless of chunk count.
#   2. The "all advantages zero" early return. advantages is a per-rank slice, so one
#      rank could skip while another entered backward. Now agreed globally.
#
# A hang is the expected failure mode, so the 1h limit IS the assertion: if this job
# times out instead of printing DONE, the collective structure is still wrong.
#
# Batch math: generation_batch_size(16) = per_device_train_batch_size(8) x 2 GPUs, and
# 8 % num_generations(8) == 0. timestep_batch_size 8 forces many chunks per step, which
# is the whole point -- with one chunk nothing is skipped and the bug cannot appear.

set -uo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/smoke_ddp

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

rm -rf "$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"
cd "$REPO"

echo "=== GPUs ==="; nvidia-smi --query-gpu=name,memory.total --format=csv

# A hang here is a claim about a specific version of the collective structure, so the
# log has to pin down which one. The short hash is not enough: the fix lives in the
# working tree until it is committed, and a job launched from a dirty tree tests code
# that no commit contains. Print the full hash and the tracked diff (names, then the
# patch) so the log alone says exactly what ran.
echo "=== commit ==="; git -C "$REPO" log -1 --format='%H%n%ci%n%s'
echo "=== tracked diff vs HEAD ==="
if git -C "$REPO" diff --quiet HEAD --; then
  echo "clean: tracked files match HEAD"
else
  echo "DIRTY: tracked files differ from HEAD -- this run is NOT commit $(git -C "$REPO" rev-parse HEAD)"
  git -C "$REPO" diff --stat HEAD --
  echo "--- patch ---"
  git -C "$REPO" diff HEAD --
  echo "--- end patch ---"
fi

accelerate launch --num_processes 2 --main_process_port 29517 \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 8 \
  --generation_batch_size 16 \
  --timestep_batch_size 8 \
  --max_steps 6 \
  --save_steps 500 \
  --warmup_steps 2
rc=$?
echo "RESULT smoke_ddp: exit=$rc"
[ $rc -eq 0 ] && echo "DONE: 2-GPU chunked backward completed without a collective hang"
