#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --job-name=eval-blocksize
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Evaluates the learned joint (block size, threshold) policy against the baselines
# that formulation.md 8 requires but Phase 0 did not cover: the AdaBlock heuristic,
# a uniform-random action, and the untrained policy.
#
# The fixed-action constant grid is already measured -- see
# eval_results/blocksweep. The bar is (b=64, tau=0.7): 79.61% at 47.2 NFE, or
# E[R] = 0.7238 at alpha=0.5.
#
# All 4 GPUs on purpose: eval.pipeline needs >=2 processes (eval.py's sampler calls
# dist.get_world_size unconditionally) and hardcodes rendezvous port 29500, so two
# jobs must never share a node.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_blocksize
RESULTS=$WORK/eval_results/llada8b_blocksize
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# 1) Learned policy. --remasking block_policy makes eval.pipeline drive the joint
#    action path; block length and threshold both come from the policy, so no
#    --block_length / --thres here. Results land in a *_blockpolicy directory and
#    record block_length="cat" so they cannot merge with fixed-block rows.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42,43,44 \
  --temperatures 1.0 \
  --remasking block_policy \
  --save_path "$RESULTS" \
  --no_aggregate

# 2) Untrained policy (formulation.md 9): must land near the random-action
#    baseline. A mismatch means the feasibility mask or head init is wrong.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints first \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --remasking block_policy \
  --save_path "$RESULTS" \
  --no_aggregate

# 3) AdaBlock heuristic at the best constant block length, for reference.
python -m eval.pipeline "baseline-fastdllm-t0.7" "$CONFIG" \
  --checkpoints self \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --block_length 64 \
  --adaptive_block --delimiter_threshold 0.3 \
  --save_path "$RESULTS" \
  --no_aggregate

# 4) Aggregate together with the Phase 0 constant grid so the learned policy is
#    scored against every constant action in its own action space.
python -m eval.aggregate_results \
  --results_dir "$RESULTS" \
  --results_dir "$WORK/eval_results/blocksweep" \
  --output_dir "$RESULTS"
