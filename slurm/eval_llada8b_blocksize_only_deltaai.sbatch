#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=08:00:00
#SBATCH --job-name=eval-blocksize-only-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Evaluates the block-size-only policy (tau pinned at 0.9) against the arms it
# has to beat. Only two arms are actually run here: the learned policy and the
# untrained control. The fixed-block constants and the AdaBlock heuristic were
# already measured at tau=0.9 and are pulled into the aggregation instead of
# being recomputed -- see the --results_dir list in step 3.
#
# Bar at alpha=1.0 (GSM8K 1319, greedy, seed 42):
#   fixed b=64       80.67% @ 78.0 NFE  E[R] = 0.5734   <- the bar
#   AdaBlock ada0.3  79.83% @ 85.3 NFE  E[R] = 0.5446
#   per-example oracle over the 5 measured b            0.6315 (+10.1%)
#
# ghx4 gives 4 GPUs per node, which is exactly what eval.pipeline needs: it
# launches with torch.cuda.device_count() processes, eval.py's sampler calls
# dist.get_world_size() unconditionally (so >=2 processes), and the rendezvous
# port 29500 is hardcoded (so the job must own the whole node).
#
# Home dirs are NOT shared with Delta, so the repo and the conda env must exist
# on this side; /work/hdd IS shared, so the checkpoints read and the results
# written here are the same ones the Delta scripts use -- run this on either
# cluster, not both at once against the same RESULTS dir.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_blocksize_only
RESULTS=$WORK/eval_results/llada8b_blocksize_only
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_blocksize_only.yaml

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# 1) Learned policy. --remasking block_policy drives the categorical action path;
#    the block length comes from the policy and the threshold is the single
#    candidate 0.9, so no --block_length / --thres here. Results land in a
#    *_blockpolicy directory and record block_length="cat" so they cannot merge
#    with fixed-block rows.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42,43,44 \
  --temperatures 1.0 \
  --remasking block_policy \
  --save_path "$RESULTS" \
  --no_aggregate

# 2) Untrained policy (formulation.md 9): must land near a uniform-random-action
#    baseline. A mismatch means the feasibility mask or the head init is wrong.
#    Given that the joint run converged to a dominated constant, this control is
#    what separates "learned nothing" from "learned something bad".
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints first \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --remasking block_policy \
  --save_path "$RESULTS" \
  --no_aggregate

# 3) Aggregate against the already-measured tau=0.9 arms. Nothing here is
#    recomputed: blocksweep holds b in {8,16,64}, the remaining two constants
#    live in the BL32/BL128 run dirs, and llada8b_bl32_adablock holds the
#    AdaBlock (ada0.3) arm. Rows are keyed by run name and block_length, so the
#    learned policy ("cat"), the constants (8/16/32/64/128) and AdaBlock
#    ("ada0.3") stay distinct.
python -m eval.aggregate_results \
  --results_dir "$RESULTS" \
  --results_dir "$WORK/eval_results/blocksweep" \
  --results_dir "$WORK/eval_results/llada8b_bl32" \
  --results_dir "$WORK/eval_results/llada8b_bl128" \
  --results_dir "$WORK/eval_results/llada8b_bl32_adablock" \
  --output_dir "$RESULTS"

# 4) Per-block action breakdown (block-size marginals per decision), which
#    aggregate_results collapses.
python -m eval.analyze_block_policy "$RESULTS" --csv "$RESULTS/block_actions.csv"

# 5) The state-dependence test (formulation.md 9, "weak-positive trap"): a policy
#    that beats the bar with ~zero mutual information between the confidence
#    field and its action found a better *marginal*, not state-dependence. The
#    joint run scored I/ln|A| ~ 0.10; that is the number to beat. max_decisions
#    is raised from the default 8 because b=8 lets a rollout reach 32 blocks.
python -m eval.analyze_action_logits "$RESULTS" \
  --alpha 1.0 \
  --block_size_candidates 8,16,32,48,64,96,128 \
  --threshold_candidates 0.9 \
  --max_decisions 12 \
  --output "$RESULTS/action_logit_analysis.txt"
