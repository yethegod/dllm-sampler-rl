#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-bu-cond-fixedb-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Does v1's unmask head (llada8b_block_unmask_cond, ckpt-1870, trained under
# near-uniform random block sizes with a window-conditioned trunk) decode
# differently depending on the block size it is given?
#
# Stage A: freeze everything, force a constant block b in {8..128}, seed 42.
#          -> acc / NFE per b with the SAME unmask head; per-question oracle over b
#             is the ceiling any stage-2 block head could reach on this decoder.
# Stage B: keep the real block fixed but lie to the window-conditioned head about it
#          (--block_unmask_cond_block). Identical behaviour == the head never learned
#          p(u | s, b) (its window_embedding norm is 0.044 vs 16.5 for mask_embedding).
# Stage C: seeds 43, 44 of stage A -> is a question's best b stable across seeds?
# Stage D: aggregate + eval.analyze_fixed_block_sweep (CPU).
#
# Every cell records unmask_order so tokens/step and frontier-distance profiles per
# b come out of the JSON. Usage: sbatch <this file> [A|B|C|D|all]  (default all);
# a rerun overwrites (eval.pipeline does not skip finished cells), so pass only the
# stages still missing. ~10-20 min per cell on 4 GPUs -> all four stages ~7h.

set -euo pipefail
STAGES=${1:-all}

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond
RESULTS=$WORK/eval_results/llada8b_block_unmask_cond_fixedb
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond.yaml

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

want() { [[ "$STAGES" == all || "$STAGES" == *"$1"* ]]; }
common=(--checkpoints last --datasets gsm8k --temperatures 1.0
        --remasking block_unmask_policy --sampling_mode bernoulli-argmax
        --block_sampling_mode categorical-argmax --record_unmask_order
        --save_path "$RESULTS" --no_aggregate)

if want A; then
  for b in 8 16 32 48 64 96 128; do
    echo "=== stage A: fixed b=$b seed 42 $(date)"
    python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --seeds 42 \
      --block_unmask_fixed_schedule "$b" "${common[@]}"
  done
fi

if want B; then
  # real b / told b: 128/8, 128/32, 32/128, 32/8, 8/128
  for pair in 128:8 128:32 32:128 32:8 8:128; do
    b=${pair%%:*}; c=${pair##*:}
    echo "=== stage B: fixed b=$b told $c seed 42 $(date)"
    python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --seeds 42 \
      --block_unmask_fixed_schedule "$b" --block_unmask_cond_block "$c" "${common[@]}"
  done
fi

if want C; then
  for b in 8 16 32 48 64 96 128; do
    echo "=== stage C: fixed b=$b seeds 43,44 $(date)"
    python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --seeds 43,44 \
      --block_unmask_fixed_schedule "$b" "${common[@]}"
  done
fi

if want D; then
  echo "=== stage D: aggregate + analysis $(date)"
  python -m eval.aggregate_results \
    --results_dir "$RESULTS" \
    --results_dir "$WORK/eval_results/llada8b_block_unmask_cond" \
    --results_dir "$WORK/eval_results/blocksweep" \
    --output_dir "$RESULTS"
  python -m eval.analyze_fixed_block_sweep "$RESULTS" \
    --learned_dir "$WORK/eval_results/llada8b_block_unmask_cond" \
    --out "$RESULTS/fixed_block_sweep_report.txt"
fi
echo "=== done $(date)"
