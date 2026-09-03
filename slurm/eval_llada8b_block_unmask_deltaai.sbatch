#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=08:00:00
#SBATCH --job-name=eval-block-unmask-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# DeltaAI eval of the joint (block size, per-position unmasking) policy trained
# by train_llada8b_block_unmask_deltaai.sbatch.
#
# ghx4 gives 4 GPUs per node, which is exactly what eval.pipeline needs: it
# launches with torch.cuda.device_count() processes, eval.py's sampler calls
# dist.get_world_size() unconditionally (so >=2 processes), and the rendezvous
# port 29500 is hardcoded (so the job must own the whole node).
#
# /work/hdd is shared with Delta -- do not run a Delta twin against the same
# RESULTS dir at the same time.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_block_unmask
RESULTS=$WORK/eval_results/llada8b_block_unmask
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint.yaml

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# 1) Learned policy, greedy deployment on both heads: bernoulli-argmax for the
#    unmask head (the paper's eval mode; a step that draws nothing is forced to
#    its argmax so blocks always finish) and categorical-argmax for the block
#    size. Results land in *_blockunmask_block_categorical-argmax and record
#    block_length="blockunmask" so they never merge with fixed-block rows.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42,43,44 \
  --temperatures 1.0 \
  --remasking block_unmask_policy \
  --sampling_mode bernoulli-argmax \
  --block_sampling_mode categorical-argmax \
  --save_path "$RESULTS" \
  --no_aggregate

# 2) Same checkpoint with the block size sampled instead of argmax'd -- the
#    E_{a~pi}[R] that GRPO actually optimised.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42,43,44 \
  --temperatures 1.0 \
  --remasking block_unmask_policy \
  --sampling_mode bernoulli-argmax \
  --block_sampling_mode categorical \
  --save_path "$RESULTS" \
  --no_aggregate

# 3) Untrained policy (formulation.md 9): uniform block sizes and the paper's
#    initial unmask rate. This is the floor the trained checkpoint must clear.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints first \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --remasking block_unmask_policy \
  --sampling_mode bernoulli-argmax \
  --block_sampling_mode categorical-argmax \
  --save_path "$RESULTS" \
  --no_aggregate

# 4) Aggregate next to the paper's BL32/BL128 per-position policies and the
#    fixed-(b, tau) grid. Rows are keyed by run name and block_length, so the
#    learned policy ("blockunmask") stays distinct from every constant.
python -m eval.aggregate_results \
  --results_dir "$RESULTS" \
  --results_dir "$WORK/eval_results/blocksweep" \
  --results_dir "$WORK/eval_results/llada8b_bl32" \
  --results_dir "$WORK/eval_results/llada8b_bl128" \
  --output_dir "$RESULTS"

# 5) Per-block action breakdown (block-size marginal per decision index).
python -m eval.analyze_block_policy "$RESULTS" --csv "$RESULTS/block_actions.csv"
