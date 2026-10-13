#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-kodcode-a1-stoch-gh
#SBATCH --output=slurm/logs/%x_%j.out

# Stochastic decoding of the alpha=1 KodCode DPLS block_unmask policy (checkpoint-1064):
# dpls unmask sampling + categorical block sampling at policy temperature 1, seeds 42-44,
# HumanEval + MBPP 0-shot. Writes next to the greedy cells (the dir name carries the
# sampling mode and seed; aggregation groups on decode_mode). DS=humaneval|mbpp selects
# the dataset so each job stays inside the 2 h interactive limit.

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
    --few_shot 0 --seeds 42,43,44 --temperatures 1.0 --remasking block_unmask_policy \
    --sampling_mode dpls --block_sampling_mode categorical \
    --save_path "$2" --no_aggregate "${EXTRA[@]}"
}

case "${DS:?set DS=humaneval or mbpp}" in
  humaneval) policy humaneval "$WORK/eval_results/code/a1_dpls_kodcode" ;;
  mbpp) policy mbpp "$WORK/eval_results/code_mbpp0shot/a1_dpls_kodcode" ;;
  *) echo "unknown DS $DS"; exit 2 ;;
esac
echo "=== done $(date) ==="
