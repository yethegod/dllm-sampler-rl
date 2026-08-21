#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --job-name=eval-bl32-fs5
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# GSM8K 5-shot rerun of the BL32 comparison (the 0-shot version is
# eval_llada8b_bl32.sbatch -> eval_results/llada8b_bl32).
#
# Three things differ from the 0-shot job and all three matter:
#
# 1. THE SHOTS ARE SEED-DEPENDENT. data/loaders/gsm8k.py draws the 5
#    demonstrations with the global RNG, which eval.py seeds from --seed. So a
#    few-shot baseline is no longer deterministic across seeds the way the
#    0-shot one is: the prompt itself changes. Hence every arm here, baselines
#    included, runs all three seeds -- the spread now covers demo choice, and
#    at a given seed every arm sees the identical prompt.
#
# 2. THE PROMPT IS ~6x LONGER (947 tokens vs 148), so each forward pass costs
#    roughly 3x more. NFE is unchanged by construction -- it counts forward
#    passes, not tokens -- so the accuracy/NFE plane stays comparable with the
#    0-shot numbers; wall time does not.
#
# 3. ONE (ARM, SEED) PER TASK, on 4-GPU nodes. Same two placement constraints as
#    eval_blocksize_sweep.sbatch: never 1 GPU (CustomDistributedSampler calls
#    dist.get_world_size() unconditionally) and claim all 4 GPUs on the node
#    (eval.pipeline hardcodes the default rendezvous port 29500, so two tasks on
#    one node collide).
#
# A shared --save_path is safe here: run_eval's directory name carries the
# checkpoint, the seed and the "_fs5" suffix, so no two tasks write the same dir.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_bl32
RESULTS=$WORK/eval_results/llada8b_bl32_fs5
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml

# (arm, seed). "policy" is the trained BL32 policy at its last checkpoint;
# the rest are heuristic baselines resolved from their names by
# eval/eval.py:parse_baseline_checkpoint.
GRID=(
  "policy 42"                       "policy 43"                       "policy 44"
  "baseline-fastdllm-t0.9 42"       "baseline-fastdllm-t0.9 43"       "baseline-fastdllm-t0.9 44"
  "baseline-low_confidence-K256 42" "baseline-low_confidence-K256 43" "baseline-low_confidence-K256 44"
  "baseline-low_confidence-K128 42" "baseline-low_confidence-K128 43" "baseline-low_confidence-K128 44"
)
read -r ARM SEED <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota. The 5-shot demos come
# from the GSM8K *train* split, which is pulled through this same cache.
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

echo "=== task $SLURM_ARRAY_TASK_ID: arm=$ARM seed=$SEED fs=5 -> $RESULTS ==="

if [ "$ARM" = "policy" ]; then
  python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
    --checkpoints last \
    --datasets gsm8k \
    --seeds "$SEED" \
    --temperatures 1.0 \
    --sampling_mode bernoulli-argmax \
    --few_shot 5 \
    --save_path "$RESULTS" \
    --no_aggregate
else
  python -m eval.pipeline "$ARM" "$CONFIG" \
    --checkpoints self \
    --datasets gsm8k \
    --seeds "$SEED" \
    --temperatures 1.0 \
    --few_shot 5 \
    --save_path "$RESULTS" \
    --no_aggregate
fi

echo "=== done: arm=$ARM seed=$SEED ==="

# Aggregate once the whole array has finished:
#   python -m eval.aggregate_results \
#     --results_dir $WORK/eval_results/llada8b_bl32_fs5 \
#     --results_dir $WORK/eval_results/llada8b_bl32
# summary_statistics.csv groups on few_shot, so the 0-shot and 5-shot rows for
# the same arm land side by side instead of averaging together.
