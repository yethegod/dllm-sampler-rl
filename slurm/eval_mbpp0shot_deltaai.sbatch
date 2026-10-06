#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=mbpp0shot
#SBATCH --output=slurm/logs/%x_%j.out

# Diagnostic: is the KodCode policy's MBPP gap a prompt-distribution shift? MBPP is the
# only 3-shot benchmark (lm-eval mbpp_instruct convention); the policy was trained on
# 0-shot chat prompts. Two cells at --few_shot 0: KodCode a1 (ckpt last, dpls-greedy)
# and its closest Fast-dLLM partner b64 tau0.7 (3-shot: 40.6 @ 48.4 vs policy 37.2 @ 45.4).
# Deterministic, seed 42. Separate RESULTS so the 3-shot cells stay untouched.

set -euo pipefail
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/code_mbpp0shot}
read -r -a EXTRA <<< "${EXTRA:-}"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + SLURM_JOB_ID % 400 * 5))
mkdir -p "$RESULTS"
cd "$REPO"

python -m eval.pipeline "$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls_kodcode" \
  configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml \
  --checkpoints last --datasets mbpp --few_shot 0 --seeds 42 --temperatures 1.0 \
  --remasking block_unmask_policy --sampling_mode dpls-greedy \
  --block_sampling_mode categorical-argmax \
  --save_path "$RESULTS/a1_dpls_kodcode" --no_aggregate "${EXTRA[@]}"

python -m eval.pipeline baseline-fastdllm-t0.7 \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --checkpoints self --datasets mbpp --few_shot 0 --seeds 42 --temperatures 1.0 \
  --block_length 64 --save_path "$RESULTS/fixed/t0.7_bl64" --no_aggregate "${EXTRA[@]}"
echo "=== done $(date) ==="
