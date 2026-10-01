#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --job-name=ada-math500-gh
#SBATCH --output=slurm/logs/%x_%A_%a.out

# AdaBlock-dLLM (arXiv:2509.26432) on MATH500 under OUR prompt (0-shot chat, L=256, no
# KV cache), at the paper's Fast-dLLM threshold 0.9 and tau_D 0.3, so it sits next to
# the alpha=1 DPLS block_unmask policy (35.4 @ 77.3) and the MATH Fast-dLLM rows.
#
#   0-2  AdaBlock t0.9, B0 = 16 / 32 / 64
#   3-4  fixed-block Fast-dLLM t0.9 at b = 16 / 64 (b = 32 already exists in
#        eval_results/llada8b_block_unmask_cond_v2_a1_dpls/baselines/t0.9_bl32), so each
#        Ada cell has its same-B0 fixed-block partner, the comparison the paper makes.
#
# Flag route (--adaptive_block + --block_length), not "baseline-...-ada0.3": see
# eval_adablock_b0_sweep_deltaai.sbatch for why the name route collides B0 cells.
# Deterministic (temperature 0, no policy, no subsampling), so seed 42 only.
# 2 GPUs per task: never 1 (eval.py's sampler needs a process group).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/adablock_math500}
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
read -r -a EXTRA <<< "${EXTRA:-}"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 400 * 5 + ${SLURM_ARRAY_TASK_ID:-0}))

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"
TASK=${TASK:-$SLURM_ARRAY_TASK_ID}
echo "=== task $TASK on $(hostname) $(date) port=$EVAL_MAIN_PROCESS_PORT ==="

ada() {  # ada <B0>
  python -m eval.pipeline baseline-fastdllm-t0.9 "$CONFIG" --checkpoints self --datasets math \
    --seeds 42 --temperatures 1.0 --block_length "$1" \
    --adaptive_block --delimiter_threshold 0.3 \
    --save_path "$RESULTS/ada" --no_aggregate "${EXTRA[@]}"
}
fixed() {  # fixed <b>; own save_path: the baseline dir name has no block length
  python -m eval.pipeline baseline-fastdllm-t0.9 "$CONFIG" --checkpoints self --datasets math \
    --seeds 42 --temperatures 1.0 --block_length "$1" \
    --save_path "$RESULTS/fixed/t0.9_bl$1" --no_aggregate "${EXTRA[@]}"
}

case "$TASK" in
  0) ada 16 ;;
  1) ada 32 ;;
  2) ada 64 ;;
  3) fixed 16 ;;
  4) fixed 64 ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== task $TASK done $(date) ==="
