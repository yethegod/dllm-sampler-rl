#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --array=0-55
#SBATCH --job-name=posthr-gh
#SBATCH --output=slurm/logs/%x_%A_%a.out

# Distilled training-free sampler vs Fast-dLLM across block sizes (GSM8K + MATH500, our
# 0-shot prompt, L=256, greedy). The learned unmask head's STOP crossing rises along the
# answer (0.57/0.70/0.76/0.81 per 64-token quarter on both benchmarks), so the heuristic
# is Fast-dLLM with tau(i) = t + s * ((i + 0.5)/L - 0.5): baseline-fastdllm-t<t>-s<s>.
# s = 0.32 is read off the learned curve (t = 0.7 gives 0.54 -> 0.86), not tuned; the
# sweep is over the mean t, the same grid as Fast-dLLM's constant threshold, so every
# heuristic cell has a constant-threshold partner at equal mean.
#
#   0-23   rising   s = +0.32   {gsm8k, math} x t {0.5, 0.7, 0.9} x b {16, 32, 64, 128}
#   24-31  Fast-dLLM (s = 0) MATH500 cells not run yet: t0.5 b16/32/64/128,
#          t0.7 b16/32/128, t0.9 b128 (t0.7 b64, t0.9 b16/32/64 exist elsewhere;
#          GSM8K's full grid is in eval_results/blocksweep)
#   32-55  falling  s = -0.32   same grid: the direction control (CadLLM's schedule decays)
#
# Deterministic (temperature 0, no policy), so seed 42 only. 2 GPUs per task, never 1
# (eval.py's sampler needs a process group). The baseline dir name carries no block
# length, so each b gets its own save_path. Run a cell by hand with TASK=<i>.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/pos_threshold}
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
read -r -a EXTRA <<< "${EXTRA:-}"

CELLS=()  # "dataset slope t b"; slope "" = constant threshold
for s in 0.32; do for ds in gsm8k math; do for t in 0.5 0.7 0.9; do for b in 16 32 64 128; do
  CELLS+=("$ds $s $t $b"); done; done; done; done
for tb in "0.5 16" "0.5 32" "0.5 64" "0.5 128" "0.7 16" "0.7 32" "0.7 128" "0.9 128"; do
  CELLS+=("math - $tb"); done
for s in -0.32; do for ds in gsm8k math; do for t in 0.5 0.7 0.9; do for b in 16 32 64 128; do
  CELLS+=("$ds $s $t $b"); done; done; done; done

TASK=${TASK:-$SLURM_ARRAY_TASK_ID}
read -r DS S T B <<< "${CELLS[$TASK]}"
NAME=baseline-fastdllm-t$T
[ "$S" != "-" ] && NAME=$NAME-s$S

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=${EVAL_MAIN_PROCESS_PORT:-$((29500 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 400 * 5 + TASK % 5))}

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"
echo "=== task $TASK: $NAME $DS b=$B on $(hostname) $(date) port=$EVAL_MAIN_PROCESS_PORT ==="

python -m eval.pipeline "$NAME" "$CONFIG" --checkpoints self --datasets "$DS" \
  --seeds 42 --temperatures 1.0 --block_length "$B" \
  --save_path "$RESULTS/$DS/bl$B" --no_aggregate "${EXTRA[@]}"

echo "=== task $TASK done $(date) ==="
