#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=mbpp0shot-rest
#SBATCH --output=slurm/logs/%x_%j.out

# MBPP at --few_shot 0 for every method in the paper (MBPP joins the other benchmarks at
# 0-shot). eval_mbpp0shot_deltaai.sbatch already ran KodCode a1 and Fast-dLLM b64 t0.7;
# these are the other 12 cells, in three lanes balanced by NFE (~1h each on 2 GH200s):
#   LANE=0  KodCode a0.5; Fast-dLLM b16 t0.7, t0.9
#   LANE=1  math a1; AdaBlock B0=32 t0.9; Fast-dLLM b32 t0.9, b64 t0.9
#   LANE=2  CadLLM; Fast-dLLM b16 t0.5, b32 t0.5, b32 t0.7, b64 t0.5
# Same results tree as the first two cells. Deterministic, seed 42.

set -euo pipefail
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/code_mbpp0shot}
BASE_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
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
COMMON=(--datasets mbpp --few_shot 0 --seeds 42 --temperatures 1.0 --no_aggregate)

fd() {  # fd <b> <tau>
  python -m eval.pipeline "baseline-fastdllm-t$2" "$BASE_CONFIG" --checkpoints self \
    --block_length "$1" --save_path "$RESULTS/fixed/t$2_bl$1" "${COMMON[@]}" "${EXTRA[@]}"
}
policy() {  # policy <run dir> <config> <ckpt> <subdir>
  python -m eval.pipeline "$WORK/checkpoints/$1" "configs/experiment_configs/$2" \
    --checkpoints "$3" --remasking block_unmask_policy --sampling_mode dpls-greedy \
    --block_sampling_mode categorical-argmax --save_path "$RESULTS/$4" "${COMMON[@]}" "${EXTRA[@]}"
}

echo "=== lane $LANE on $(hostname) $(date)"
case "$LANE" in
  0) policy llada8b_block_unmask_cond_v2_a0p5_dpls_kodcode \
       llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0p5_dpls_kodcode.yaml last a0p5_dpls_kodcode
     fd 16 0.7; fd 16 0.9 ;;
  1) policy llada8b_block_unmask_cond_v2_a1_dpls \
       llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls.yaml 1870 a1_dpls
     python -m eval.pipeline baseline-fastdllm-t0.9 "$BASE_CONFIG" --checkpoints self \
       --block_length 32 --adaptive_block --delimiter_threshold 0.3 \
       --save_path "$RESULTS/ada" "${COMMON[@]}" "${EXTRA[@]}"
     fd 32 0.9; fd 64 0.9 ;;
  2) python -m eval.pipeline baseline-cadllm "$BASE_CONFIG" --checkpoints self \
       --save_path "$RESULTS/cadllm" "${COMMON[@]}" "${EXTRA[@]}"
     fd 16 0.5; fd 32 0.5; fd 32 0.7; fd 64 0.5 ;;
  *) echo "unknown LANE $LANE"; exit 2 ;;
esac
echo "=== lane $LANE done $(date)"
