#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-kodcode-a1-nowin-gh
#SBATCH --output=slurm/logs/%x_%j.out

# Window-conditioning ablation on the code side, the twin of the JNOWIN cell of
# eval_ablation_block_rule_deltaai.sbatch: the alpha=1 KodCode DPLS block_unmask policy
# (checkpoint-1064) with its window embedding zeroed (--zero_window_cond), so the unmask
# head is not told its block while the learned block head is unchanged. HumanEval +
# MBPP, 0-shot, greedy (dpls-greedy unmask + categorical-argmax block), seed 42.
# Compare with the learned cells in code/a1_dpls_kodcode and code_mbpp0shot/a1_dpls_kodcode
# (HumanEval 41.5 @ 71.5, MBPP 39.4 @ 49.9). 2 GPUs (eval.py's sampler needs a process group).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls_kodcode
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml
# Extra pipeline flags, e.g. EXTRA="--n_test 4" for a smoke.
read -r -a EXTRA <<< "${EXTRA:-}"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + SLURM_JOB_ID % 400 * 5))

mkdir -p "$REPO/slurm/logs"
cd "$REPO"
echo "=== on $(hostname) $(date) port=$EVAL_MAIN_PROCESS_PORT ==="

policy() {  # policy <dataset> <save path>
  python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints 1064 --datasets "$1" \
    --few_shot 0 --seeds 42 --temperatures 1.0 --remasking block_unmask_policy \
    --sampling_mode dpls-greedy --block_sampling_mode categorical-argmax \
    --zero_window_cond --save_path "$2" --no_aggregate "${EXTRA[@]}"
}

policy humaneval "$WORK/eval_results/code/a1_dpls_kodcode_nowin"
policy mbpp "$WORK/eval_results/code_mbpp0shot/a1_dpls_kodcode_nowin"
echo "=== done $(date) ==="
