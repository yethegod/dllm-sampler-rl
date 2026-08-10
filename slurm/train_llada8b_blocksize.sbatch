#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=25:00:00
#SBATCH --job-name=llada8b-blocksize
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Joint (block size, Fast-dLLM threshold) policy; see formulation.md.
#
# WATCH FOR EARLY COLLAPSE (formulation.md 9). Within a few hundred steps, check
# the TensorBoard scalars block_size/frac_* and threshold/frac_*. If they converge
# to a point mass on (64, 0.7) the policy has learned the marginal -- i.e. the
# constant we already measured -- and the run should be killed rather than left to
# finish. The bar to beat at alpha=0.5 is E[R] = 0.7238 (80.7% accuracy).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_blocksize

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml \
  --output_dir "$OUTPUT_DIR"
