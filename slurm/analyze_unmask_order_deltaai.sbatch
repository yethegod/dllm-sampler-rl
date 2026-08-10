#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=00:30:00
#SBATCH --array=0-7
#SBATCH --job-name=unmask-order-gh
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

set -euo pipefail

# One short 1-GPU task per decoding configuration: eight 30-minute tasks backfill
# into gaps that a single long job never gets. Tasks are idempotent, so the array
# can simply be resubmitted to fill in whatever did not finish.
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUT=$WORK/eval_results/unmask_order_gh

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$OUT" "$REPO/slurm/logs"
cd "$REPO"

CFG32=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
CFG128=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL128_mixture.yaml
P32=$WORK/checkpoints/llada8b_bl32/checkpoint-1870/model.safetensors
P128=$WORK/checkpoints/llada8b_bl128/checkpoint-1870/model.safetensors

case ${SLURM_ARRAY_TASK_ID} in
  0) LABEL=bl32-policy;         CFG=$CFG32;  ARGS="--policy_path $P32  --sampling_mode bernoulli-argmax" ;;
  1) LABEL=bl128-policy;        CFG=$CFG128; ARGS="--policy_path $P128 --sampling_mode bernoulli-argmax" ;;
  # speed-matched controls: tau_pi moves each policy onto the other's NFE
  2) LABEL=bl32-policy-t0.5;    CFG=$CFG32;  ARGS="--policy_path $P32  --temperature_policy 0.5 --sampling_mode bernoulli-argmax" ;;
  3) LABEL=bl128-policy-t1.5;   CFG=$CFG128; ARGS="--policy_path $P128 --temperature_policy 1.5 --sampling_mode bernoulli-argmax" ;;
  # heuristic references at the same block lengths
  4) LABEL=bl32-fastdllm-t0.9;  CFG=$CFG32;  ARGS="--remasking fastdllm --thres 0.9" ;;
  5) LABEL=bl128-fastdllm-t0.9; CFG=$CFG128; ARGS="--remasking fastdllm --thres 0.9" ;;
  6) LABEL=bl32-lowconf-K256;   CFG=$CFG32;  ARGS="--remasking low_confidence --diffusion_steps 256" ;;
  7) LABEL=bl128-lowconf-K256;  CFG=$CFG128; ARGS="--remasking low_confidence --diffusion_steps 256" ;;
  *) echo "bad array id"; exit 1 ;;
esac

if [ -f "$OUT/${LABEL}_order_stats.json" ]; then
  echo "$LABEL already done, skipping"
  exit 0
fi

python -m eval.analyze_unmask_order --config "$CFG" --label "$LABEL" \
  --out "$OUT" --n_test 64 --batch_size 1 --seed 42 $ARGS
