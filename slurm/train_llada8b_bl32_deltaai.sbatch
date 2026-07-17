#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=25:00:00
#SBATCH --job-name=llada8b-bl32-gh200
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

set -euo pipefail

# DeltaAI GH200 (aarch64). Home dirs are NOT shared with Delta: the repo and
# the conda env must exist on the DeltaAI side. /work/hdd IS shared, so the
# checkpoints and HF cache below are the same ones the Delta 8-GPU run used;
# train.py auto-resumes from the latest checkpoint in OUTPUT_DIR.
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"

# .env (HF_TOKEN) is loaded from the working directory by train.py
cd "$REPO"

# per_device_train_batch_size 32 keeps the effective batch at 4*32 = 128 =
# generation_batch_size, matching the 8-GPU run (8*16) so the resumed
# training dynamics and step semantics are unchanged.
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --per_device_train_batch_size 32 \
  --output_dir "$OUTPUT_DIR"
