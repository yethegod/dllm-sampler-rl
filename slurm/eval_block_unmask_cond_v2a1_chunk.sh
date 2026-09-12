#!/bin/bash
# One chunk of the v2 alpha=1 (no stall charge) block_unmask eval (job 3119558, TIMEOUT at step 1236), sized for a 2h ghx4-interactive
# allocation on a whole node (eval.pipeline hardcodes rendezvous port 29500 and
# launches torch.cuda.device_count() processes). Same stages as
# eval_llada8b_block_unmask_cond_v2_deltaai.sbatch (v2), split so each piece fits:
#   A  checkpoint-1200 greedy (bernoulli-argmax + categorical-argmax), seeds 42,43,44
#   B  checkpoint-1200 stochastic block head (categorical), seeds 42,43,44
#   C  checkpoint-50 / 200 (pre-collapse, b=128 ~25%) / best greedy, seed 42
#   D  aggregate against v1 + blocksweep + BL32/BL128, then analyze_block_policy (CPU)
# Usage (from a login node):
#   srun --account=bhta-dtai-gh --partition=ghx4-interactive --nodes=1 \
#        --gpus-per-node=4 --cpus-per-task=32 --mem=200g --time=02:00:00 \
#        bash slurm/eval_block_unmask_cond_v2a1_chunk.sh A
set -euo pipefail
CHUNK=${1:?chunk A|B|C|D}

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1
RESULTS=$WORK/eval_results/llada8b_block_unmask_cond_v2_a1
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1.yaml

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
mkdir -p "$RESULTS"
cd "$REPO"
echo "=== chunk $CHUNK on $(hostname) $(date) ==="

common=(--datasets gsm8k --temperatures 1.0 --remasking block_unmask_policy
        --sampling_mode bernoulli-argmax --save_path "$RESULTS" --no_aggregate)
case "$CHUNK" in
  A) python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints last --seeds 42,43,44 \
       --block_sampling_mode categorical-argmax "${common[@]}" ;;
  B) python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints last --seeds 42,43,44 \
       --block_sampling_mode categorical "${common[@]}" ;;
  C) python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints first,200,best --seeds 42 \
       --block_sampling_mode categorical-argmax "${common[@]}" ;;
  D) python -m eval.aggregate_results \
       --results_dir "$RESULTS" \
       --results_dir "$WORK/eval_results/blocksweep" \
       --results_dir "$WORK/eval_results/llada8b_bl32" \
       --results_dir "$WORK/eval_results/llada8b_bl128" \
       --results_dir "$WORK/eval_results/llada8b_block_unmask_cond" \
       --results_dir "$WORK/eval_results/llada8b_block_unmask_cond_v2" \
       --output_dir "$RESULTS"
     python -m eval.analyze_block_policy "$RESULTS" --csv "$RESULTS/block_actions.csv" ;;
  *) echo "unknown chunk $CHUNK"; exit 2 ;;
esac
echo "=== chunk $CHUNK done $(date) ==="
