#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=04:00:00
#SBATCH --array=4
#SBATCH --job-name=eval-code-delta
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Delta (A100-40GB) copy of eval_code_deltaai.sbatch, made 2026-10-04 because the DeltaAI
# interactive slot for task 4 (3304765) was ~12h out. Same task numbering:
#
#   0  a1 DPLS ckpt-1870 (math-trained), greedy: dpls-greedy unmask + categorical-argmax block
#   1  Fast-dLLM t0.9 b32
#   2  AdaBlock t0.9, B0=32, tau_D 0.3
#   3  CadLLM, per-task settings
#   4  a1 DPLS trained on KodCode only, last checkpoint, same greedy modes as 0
#   5  same as 4 for the alpha=0.5 KodCode run
#
# Default array = 4 only (KodCode). Writes the same eval_results/code as the DeltaAI
# script, so do not let this and the DeltaAI task-4 job run at the same time.
# 4h limit: a GH200 task took 20-27 min on 2 GPUs; A100 is slower.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/code}
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls
POLICY_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls.yaml
BASE_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
DATASETS=${DATASETS:-humaneval,mbpp}
# Extra pipeline flags, e.g. EXTRA="--n_test 4" for a smoke.
read -r -a EXTRA <<< "${EXTRA:-}"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm
export HF_HOME=$WORK/hf_cache
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=16
export EVAL_MAIN_PROCESS_PORT=$((29500 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 400 * 5 + ${SLURM_ARRAY_TASK_ID:-${TASK:-0}}))

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"
TASK=${TASK:-$SLURM_ARRAY_TASK_ID}
echo "=== task $TASK on $(hostname) $(date) port=$EVAL_MAIN_PROCESS_PORT ==="

base() {  # base <run name> <save subdir> [pipeline flags...]
  local run=$1 sub=$2; shift 2
  python -m eval.pipeline "$run" "$BASE_CONFIG" --checkpoints self --datasets "$DATASETS" \
    --seeds 42 --temperatures 1.0 "$@" \
    --save_path "$RESULTS/$sub" --no_aggregate "${EXTRA[@]}"
}

case "$TASK" in
  0) python -m eval.pipeline "$CKPT_DIR" "$POLICY_CONFIG" --checkpoints 1870 \
       --datasets "$DATASETS" --seeds 42 --temperatures 1.0 --remasking block_unmask_policy \
       --sampling_mode dpls-greedy --block_sampling_mode categorical-argmax \
       --save_path "$RESULTS/a1_dpls" --no_aggregate "${EXTRA[@]}" ;;
  4) python -m eval.pipeline "$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls_kodcode" \
       configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml \
       --checkpoints last --datasets "$DATASETS" --seeds 42 --temperatures 1.0 \
       --remasking block_unmask_policy --sampling_mode dpls-greedy \
       --block_sampling_mode categorical-argmax \
       --save_path "$RESULTS/a1_dpls_kodcode" --no_aggregate "${EXTRA[@]}" ;;
  5) python -m eval.pipeline "$WORK/checkpoints/llada8b_block_unmask_cond_v2_a0p5_dpls_kodcode" \
       configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0p5_dpls_kodcode.yaml \
       --checkpoints last --datasets "$DATASETS" --seeds 42 --temperatures 1.0 \
       --remasking block_unmask_policy --sampling_mode dpls-greedy \
       --block_sampling_mode categorical-argmax \
       --save_path "$RESULTS/a0p5_dpls_kodcode" --no_aggregate "${EXTRA[@]}" ;;
  1) base baseline-fastdllm-t0.9 fixed/t0.9_bl32 --block_length 32 ;;
  2) base baseline-fastdllm-t0.9 ada --block_length 32 --adaptive_block --delimiter_threshold 0.3 ;;
  3) base baseline-cadllm cadllm ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
echo "=== task $TASK done $(date) ==="
