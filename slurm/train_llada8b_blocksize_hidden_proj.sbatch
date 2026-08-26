#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=48:00:00
#SBATCH --job-name=llada8b-blocksize-hproj
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Joint (block size, threshold) policy reading the dLLM's last-layer hidden state
# through a Linear(4096 -> 128) instead of top-p confidences. Same action space,
# reward, and batch math as the confidence-input run (llada8b_blocksize, job
# 20997718), so the comparison isolates the input representation.
#
# Memory: measured 28.6 GiB peak at per_device_train_batch_size 16 on a 40GB A100
# (job 21410797); the config's timestep_batch_size 8 chunks compute_loss's backward.
#
# WATCH FOR EARLY COLLAPSE (formulation.md 9). Within a few hundred steps, check
# block_size/frac_* and threshold/frac_*. The confidence-input joint run converged
# to the dominated constant (32, 0.9)/(8, 0.9) and scored E[R]=0.6803 against the
# 0.7238 bar at alpha=0.5 -- if this run collapses to a point mass the same way,
# kill it rather than let it finish.
#
# First multi-GPU exercise of block_policy x chunked backward: the DDP collective
# structure (one reducing backward per rank, f965784) was smoke-tested on the
# per-step path (job 21456347); a hang in the first few steps would falsify it here.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
# Fresh dir, never the confidence run's: resume is automatic and the hidden-proj
# policy has a different state_dict shape.
OUTPUT_DIR=$WORK/checkpoints/llada8b_blocksize_hidden_proj

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

echo "=== commit ==="; git log -1 --format='%H %s'

accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint_hidden_proj.yaml \
  --output_dir "$OUTPUT_DIR"
