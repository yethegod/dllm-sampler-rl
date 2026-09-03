#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=01:00:00
#SBATCH --job-name=smoke-block-unmask-cond-ddp-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# 6-step, 2-GPU DDP smoke for the window-conditioned block_unmask run. The 1-GPU
# smoke cannot see the two things this run adds on top of it:
#   - timestep_batch_size chunking (set in the *_cond.yaml, forced smaller here):
#     compute_loss backwards every chunk but the last under no_sync, and the chunk
#     count differs per rank with T, so only >1 rank exercises the sync structure
#     (see slurm/smoke_ddp.sbatch). A hang is the failure mode: the 1h limit IS
#     the assertion.
#   - the second trunk pass's memory at a per-device batch that is not tiny.
# Batch math: generation_batch_size(16) = per_device_train_batch_size(8) x 2 GPUs,
# and pdbs(8) % num_generations(8) == 0.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
# Override to smoke the conditionally independent control on the same DDP path:
#   sbatch --export=ALL,CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint.yaml,TAG=indep slurm/smoke_block_unmask_cond_ddp_deltaai.sbatch
CONFIG=${CONFIG:-configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond.yaml}
TAG=${TAG:-cond}
OUTPUT_DIR=$WORK/checkpoints/smoke_block_unmask_${TAG}_ddp

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16
# Force many timestep chunks per step so the no_sync structure is actually exercised.
export SMOKE_TS_BS=8

# Fresh dir every time: train.py auto-resumes from get_last_checkpoint(),
# and a stale smoke checkpoint would silently skip the steps we want to see.
rm -rf "$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

cd "$REPO"

accelerate launch --num_processes 2 --main_process_port 29531 \
  -m train.train \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 8 \
  --generation_batch_size 16 \
  --max_steps 6 \
  --save_steps 500 \
  --warmup_steps 2 \
  --timestep_batch_size "$SMOKE_TS_BS"
