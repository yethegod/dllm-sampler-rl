#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-kodcode-a3-gh
#SBATCH --output=slurm/logs/%x_%j.out

# DeltaAI twin of task 2 of eval_kodcode_a3_delta.sbatch: HumanEval + MBPP of the final
# checkpoint (1064) of the alpha=3 KodCode DPLS block_unmask run, 0-shot, greedy
# (dpls-greedy unmask + categorical-argmax block), seed 42. Same output dirs as the Delta
# script, so run only one of the two. 2 GPUs (eval.py's sampler needs a process group).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a3_dpls_kodcode
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha3_dpls_kodcode.yaml
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
  python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints last --datasets "$1" \
    --few_shot 0 --seeds 42 --temperatures 1.0 --remasking block_unmask_policy \
    --sampling_mode dpls-greedy --block_sampling_mode categorical-argmax \
    --save_path "$2" --no_aggregate "${EXTRA[@]}"
}

policy humaneval "$WORK/eval_results/code/a3_dpls_kodcode"
policy mbpp "$WORK/eval_results/code_mbpp0shot/a3_dpls_kodcode"
echo "=== done $(date) ==="
