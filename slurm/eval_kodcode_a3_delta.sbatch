#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --job-name=eval-kodcode-a3-delta
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# GSM8K / MATH500 / HumanEval / MBPP eval of the final checkpoint (1064) of the alpha=3 KodCode
# DPLS block_unmask run (train_llada8b_block_unmask_cond_v2a3_dpls_kodcode.sbatch). Everything
# at --few_shot 0, greedy (dpls-greedy unmask + categorical-argmax block), deterministic, seed 42:
#
#   0  gsm8k                 -> $WORK/eval_results/kodcode_dpls/a3_dpls_kodcode
#   1  math                  -> same
#   2  humaneval, then mbpp  -> code/a3_dpls_kodcode and code_mbpp0shot/a3_dpls_kodcode, next to
#                               the a1 / a0.5 KodCode cells and the baselines there
#
# Run times of the same policy shape on 2 A100s: gsm8k ~1.9h, math ~1.3h, humaneval+mbpp ~1h.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a3_dpls_kodcode
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha3_dpls_kodcode.yaml
# Extra pipeline flags, e.g. EXTRA="--n_test 4" for a smoke.
read -r -a EXTRA <<< "${EXTRA:-}"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm
export HF_HOME=$WORK/hf_cache
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=16
export EVAL_MAIN_PROCESS_PORT=$((29500 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 400 * 5 + ${SLURM_ARRAY_TASK_ID:-${TASK:-0}}))

mkdir -p "$REPO/slurm/logs"
cd "$REPO"
TASK=${TASK:-$SLURM_ARRAY_TASK_ID}
echo "=== task $TASK on $(hostname) $(date) port=$EVAL_MAIN_PROCESS_PORT ==="

policy() {  # policy <dataset> <save path>
  python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints last --datasets "$1" \
    --few_shot 0 --seeds 42 --temperatures 1.0 --remasking block_unmask_policy \
    --sampling_mode dpls-greedy --block_sampling_mode categorical-argmax \
    --save_path "$2" --no_aggregate "${EXTRA[@]}"
}

case "$TASK" in
  0) policy gsm8k "$WORK/eval_results/kodcode_dpls/a3_dpls_kodcode" ;;
  1) policy math "$WORK/eval_results/kodcode_dpls/a3_dpls_kodcode" ;;
  2) policy humaneval "$WORK/eval_results/code/a3_dpls_kodcode"
     policy mbpp "$WORK/eval_results/code_mbpp0shot/a3_dpls_kodcode" ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== task $TASK done $(date) ==="
