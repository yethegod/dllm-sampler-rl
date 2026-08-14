#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --job-name=inputprobe
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Dump the block-size policy's decision-0 observation (top-1 confidence over the
# fully masked generation region) for every GSM8K test problem, so
# eval/probe_policy_inputs.py can ask whether that observation carries any
# per-problem signal at all. See the module docstring for why decision 0 is the
# state that separates "blind policy" from "blind observation".
#
# One GPU is fine here: this is a standalone script, not eval.pipeline, so
# neither the CustomDistributedSampler world-size assumption nor the hardcoded
# rendezvous port applies.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUT=$WORK/eval_results/input_probe
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$OUT" "$REPO/slurm/logs"
cd "$REPO"

python -m eval.probe_policy_inputs --mode dump \
  --config "$CONFIG" \
  --dataset gsm8k \
  --batch_size 8 \
  --out "$OUT/c_decision0.npz"

python -m eval.probe_policy_inputs --mode analyze \
  --npz "$OUT/c_decision0.npz" \
  --sweep_dir "$WORK/eval_results/blocksweep" \
  --ref_run t0.7_bl64 \
  | tee "$OUT/probe_report.txt"

echo "=== done ==="
