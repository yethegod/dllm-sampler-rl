#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --job-name=cadllm-gh
#SBATCH --output=slurm/logs/%x_%A_%a.out

# CadLLM (arXiv:2512.07173) under OUR prompt (0-shot chat, L=256, no KV cache), ported
# into generate_unified as remasking=cadllm with its released GSM8K/MATH settings (see
# common/generation/cadllm.py; tests/test_cadllm.py matches the original token for
# token). Sits next to the alpha=1 DPLS block_unmask policy, AdaBlock and the
# fixed-block Fast-dLLM rows.
#
#   0  GSM8K
#   1  MATH500
#
# GH200 only: greedy decodes differ between A100 and GH200, keep one GPU type per table.
# Deterministic (temperature 0, no policy, no subsampling), so seed 42 only.
# 2 GPUs per task: never 1 (eval.py's sampler needs a process group).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/cadllm}
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

cad() {  # cad <dataset>
  python -m eval.pipeline baseline-cadllm "$CONFIG" --checkpoints self --datasets "$1" \
    --seeds 42 --temperatures 1.0 \
    --save_path "$RESULTS" --no_aggregate "${EXTRA[@]}"
}

case "$TASK" in
  0) cad gsm8k ;;
  1) cad math ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== task $TASK done $(date) ==="
