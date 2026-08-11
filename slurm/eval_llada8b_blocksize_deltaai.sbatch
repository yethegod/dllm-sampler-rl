#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=08:00:00
#SBATCH --job-name=eval-blocksize-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# DeltaAI (GH200, aarch64) port of eval_llada8b_blocksize.sbatch. Home dirs are
# NOT shared with Delta, so the repo and the conda env must exist on this side;
# /work/hdd IS shared, so the checkpoints read and the results written here are
# the same ones the Delta scripts use -- run this on either cluster, not both at
# once against the same RESULTS dir.
#
# ghx4 gives 4 GPUs per node, which is exactly what eval.pipeline needs: it
# launches with torch.cuda.device_count() processes, eval.py's sampler calls
# dist.get_world_size() unconditionally (so >=2 processes), and the rendezvous
# port 29500 is hardcoded (so the job must own the whole node).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_blocksize
RESULTS=$WORK/eval_results/llada8b_blocksize
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
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

# 5) Per-block action breakdown (block-size/threshold marginals + joint), which
#    aggregate_results collapses.
python -m eval.analyze_block_policy "$RESULTS" --csv "$RESULTS/block_actions.csv"
