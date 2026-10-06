#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=fd-grid-gh
#SBATCH --output=slurm/logs/%x_%A_%a.out

# Fixed-block Fast-dLLM grid b in {16,32,64} x tau in {0.5,0.7,0.9} on MATH500,
# HumanEval and MBPP, so the alpha=1 policy can be set against the hand-tuned
# Fast-dLLM envelope on every benchmark (GSM8K already has all 9 cells, in
# eval_results/blocksweep). Only the cells that do not exist yet:
#
#   0  code  b32  tau 0.5, 0.7
#   1  math  b16  tau 0.5, 0.7
#   2  math  b32  tau 0.5, 0.7
#   3  math  b64  tau 0.5
#   4  code  b16  tau 0.5, 0.7, 0.9
#   5  code  b64  tau 0.5, 0.7, 0.9
#
# Existing MATH500 cells: b16/b64 t0.9 in adablock_math500/fixed, b32 t0.9 and b64 t0.7
# in llada8b_block_unmask_cond_v2_a1_dpls/baselines. Existing code cell: b32 t0.9 in
# code/fixed. Same prompt and config as those (0-shot chat; MBPP 3-shot, L=256, no KV
# cache). MBPP uses the fixed-fence sanitize_mbpp (3916cef). Deterministic, seed 42 only.
# One pipeline call per tau, each in its own t<tau>_bl<b> dir (the baseline dir name has
# no block length). Every task fits the 2h ghx4-interactive cap:
#   srun --account=bhta-dtai-gh --partition=ghx4-interactive --gpus-per-node=2 ... \
#     env TASK=<n> bash slurm/eval_fd_grid_deltaai.sbatch
# 2 GPUs per task: never 1 (eval.py's sampler needs a process group).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
MATH_RESULTS=${MATH_RESULTS:-$WORK/eval_results/fd_grid_math500}
CODE_RESULTS=${CODE_RESULTS:-$WORK/eval_results/code/fixed}
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
# Extra pipeline flags, e.g. EXTRA="--n_test 4" for a smoke.
read -r -a EXTRA <<< "${EXTRA:-}"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 400 * 5 + ${SLURM_ARRAY_TASK_ID:-${TASK:-0}}))

mkdir -p "$MATH_RESULTS" "$CODE_RESULTS" "$REPO/slurm/logs"
cd "$REPO"
TASK=${TASK:-$SLURM_ARRAY_TASK_ID}
echo "=== task $TASK on $(hostname) $(date) port=$EVAL_MAIN_PROCESS_PORT ==="

fd() {  # fd <datasets> <results root> <b> <tau>...
  local ds=$1 root=$2 b=$3; shift 3
  for t in "$@"; do
    echo "--- $ds b$b t$t $(date)"
    python -m eval.pipeline "baseline-fastdllm-t$t" "$CONFIG" --checkpoints self --datasets "$ds" \
      --seeds 42 --temperatures 1.0 --block_length "$b" \
      --save_path "$root/t${t}_bl$b" --no_aggregate "${EXTRA[@]}"
  done
}

case "$TASK" in
  0) fd humaneval,mbpp "$CODE_RESULTS" 32 0.5 0.7 ;;
  1) fd math "$MATH_RESULTS" 16 0.5 0.7 ;;
  2) fd math "$MATH_RESULTS" 32 0.5 0.7 ;;
  3) fd math "$MATH_RESULTS" 64 0.5 ;;
  4) fd humaneval,mbpp "$CODE_RESULTS" 16 0.5 0.7 0.9 ;;
  5) fd humaneval,mbpp "$CODE_RESULTS" 64 0.5 0.7 0.9 ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== task $TASK done $(date) ==="
