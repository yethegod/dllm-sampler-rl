#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --job-name=eval-bu-a1-dpls-gh
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# GSM8K + MATH500 eval of the FINAL checkpoint of the alpha=1 DPLS block_unmask run
# (train_llada8b_block_unmask_cond_v2a1_dpls_deltaai.sbatch, job 3218245).
# Submit with --dependency=afterok:<train job> so `--checkpoints last` is the end of training.
#
#   0  final, greedy:     dpls-greedy unmask + categorical-argmax block, gsm8k, seeds 42,43,44
#   1  final, stochastic: dpls unmask + categorical block (the E[R] GRPO optimised), gsm8k, 3 seeds
#   2  final, greedy + stochastic on math, 3 seeds each
#   3  math baselines (none exist yet): Fast-dLLM t0.7 b64 (the GSM8K E[R]a1 bar) and
#      t0.9 b32 (paper default). Deterministic, seed 42.
# Aggregation: eval_llada8b_block_unmask_cond_v2a1_dpls_aggregate.sh (CPU), after the array.
#
# 2 GPUs per task backfills on ghx4 far sooner than 4; two tasks can share a node, so each
# gets its own accelerate port (EVAL_MAIN_PROCESS_PORT, read by eval/pipeline.py). Never 1
# GPU: eval.py's sampler needs a process group.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=${CKPT_DIR:-$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls}
RESULTS=${RESULTS:-$WORK/eval_results/llada8b_block_unmask_cond_v2_a1_dpls}
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls.yaml
# Baselines use the config the GSM8K blocksweep bar was computed with (same 256 gen length).
BASE_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
CKPTS=${CKPTS:-last}
SEEDS=${SEEDS:-42,43,44}
# Extra pipeline flags, e.g. EXTRA="--n_test 8" for a smoke.
read -r -a EXTRA <<< "${EXTRA:-}"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 400 * 4 + ${SLURM_ARRAY_TASK_ID:-0}))

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"
TASK=${TASK:-$SLURM_ARRAY_TASK_ID}
echo "=== task $TASK on $(hostname) $(date), ckpts=$CKPTS port=$EVAL_MAIN_PROCESS_PORT ==="

policy() {  # policy <datasets> <unmask mode> <block mode>
  python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints "$CKPTS" --datasets "$1" \
    --seeds "$SEEDS" --temperatures 1.0 --remasking block_unmask_policy \
    --sampling_mode "$2" --block_sampling_mode "$3" \
    --save_path "$RESULTS" --no_aggregate "${EXTRA[@]}"
}
baseline() {  # baseline <thres> <block_length>; own save_path: the dir name has no block length
  python -m eval.pipeline "baseline-fastdllm-t$1" "$BASE_CONFIG" --checkpoints self --datasets math \
    --seeds 42 --temperatures 1.0 --block_length "$2" \
    --save_path "$RESULTS/baselines/t$1_bl$2" --no_aggregate "${EXTRA[@]}"
}

case "$TASK" in
  0) policy gsm8k dpls-greedy categorical-argmax ;;
  1) policy gsm8k dpls categorical ;;
  2) policy math dpls-greedy categorical-argmax
     policy math dpls categorical ;;
  3) baseline 0.7 64
     baseline 0.9 32 ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== task $TASK done $(date) ==="
