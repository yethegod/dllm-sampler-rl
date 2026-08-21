#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --job-name=eval-tau09-fs5
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# The tau=0.9 block-length comparison, rerun at 5 shots: learned block-size
# policy vs AdaBlock at B0 in {16,32,64} vs the best constant b=64. Everything
# else is held fixed -- GSM8K 1319, LLaDA-8B-Instruct, L=256, greedy token
# decoding, Fast-dLLM threshold 0.9 -- so the only thing that moves between arms
# is how the block boundary is chosen.
#
# The 0-shot row of this same table (eval_results/{blocksweep,adablock_b0_sweep,
# llada8b_blocksize_only}):
#   learned policy  80.09% +/- 0.64 @ 81.0 NFE
#   AdaBlock B0=16  80.52%          @ 89.0 NFE
#   AdaBlock B0=32  79.83%          @ 85.3 NFE
#   AdaBlock B0=64  never run       <- task 15 below fills it
#   fixed b=64      80.67%          @ 78.0 NFE   <- the bar at 0 shots
#
# Three things that decide how this job is laid out:
#
# 1. AT k>0 SHOTS NOTHING IS SEED-INVARIANT. data/loaders/gsm8k.py draws the
#    demonstrations with the global RNG, seeded from --seed, so the prompt
#    itself changes with the seed. The 0-shot shortcut "AdaBlock and fixed-b are
#    deterministic, one seed is enough" does not survive: every 5-shot arm here
#    runs seeds 42/43/44, and at a given seed all five arms see the identical
#    prompt. Only the 0-shot fill-in task stays at one seed.
#
# 2. THE POLICY ARM STAYS ON THE CONFIG'S sampling_mode (categorical, i.e.
#    sampled actions), which is what produced the 80.09% above and what every
#    E[R] number in llada_8b_instruct_dit_blocksize_only.yaml is measured
#    against. It is not the greedy deployment number: for that, add
#    --sampling_mode categorical-argmax to the policy arm -- but then nothing in
#    the 0-shot row is comparable, since no argmax cell was ever run for this
#    checkpoint.
#
# 3. ADABLOCK GOES THROUGH THE FLAG ROUTE, not the name route: the baseline is
#    "baseline-fastdllm-t0.9" plus --adaptive_block --delimiter_threshold
#    --block_length. Naming it "...-ada0.3" instead would set adaptive_block
#    inside eval.py while leaving eval.pipeline's cfg.adaptive_block False, so
#    the "_ada<tau_D>_B<B0>" directory suffix would never be appended and the
#    three B0 values would overwrite each other. See eval_adablock_b0_sweep.sbatch.
#
# Placement: 4 GPUs, whole node, one (arm, seed) per task. eval.pipeline launches
# torch.cuda.device_count() processes, eval.py's sampler calls
# dist.get_world_size() unconditionally (so >=2), and the rendezvous port 29500
# is hardcoded (so two tasks must not land on one node).
#
# A shared --save_path is safe: run_eval's directory name carries the checkpoint,
# the seed, the "_fs5" suffix and the "_ada0.3_B<B0>" suffix, so no two tasks
# write the same directory.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
POLICY_CKPT=$WORK/checkpoints/llada8b_blocksize_only
RESULTS_FS5=$WORK/eval_results/tau09_fs5
# The 0-shot fill-in joins its siblings so the B0 sweep is one directory.
RESULTS_FS0=$WORK/eval_results/adablock_b0_sweep

# Architecture config for the learned policy; the baselines only take gen_length
# (256) from theirs, and this is the one the existing 0-shot cells used.
POLICY_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_blocksize_only.yaml
BASE_CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml

DELIM_THRES=0.3

# (arm, seed, shots). Tasks 0-14 are the 5-shot table; task 15 backfills the one
# missing 0-shot cell.
GRID=(
  "policy 42 5"   "policy 43 5"   "policy 44 5"
  "ada16 42 5"    "ada16 43 5"    "ada16 44 5"
  "ada32 42 5"    "ada32 43 5"    "ada32 44 5"
  "ada64 42 5"    "ada64 43 5"    "ada64 44 5"
  "fixed64 42 5"  "fixed64 43 5"  "fixed64 44 5"
  "ada64 42 0"
)
read -r ARM SEED SHOTS <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota. The demonstrations
# come from the GSM8K *train* split, pulled through this same cache.
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

if [ "$SHOTS" -gt 0 ]; then
  SAVE=$RESULTS_FS5
  FS_ARGS=(--few_shot "$SHOTS")
else
  SAVE=$RESULTS_FS0
  FS_ARGS=()
fi

mkdir -p "$SAVE" "$REPO/slurm/logs"
cd "$REPO"

echo "=== task $SLURM_ARRAY_TASK_ID: arm=$ARM seed=$SEED shots=$SHOTS -> $SAVE ==="

case "$ARM" in
  policy)
    # --remasking block_policy drives the categorical action path. The block
    # length comes from the policy and threshold_candidates is the single value
    # 0.9, so there is no --block_length / --thres to pass. Results land in a
    # *_blockpolicy directory and record block_length="cat".
    python -m eval.pipeline "$POLICY_CKPT" "$POLICY_CONFIG" \
      --checkpoints last \
      --datasets gsm8k \
      --seeds "$SEED" \
      --temperatures 1.0 \
      --remasking block_policy \
      "${FS_ARGS[@]}" \
      --save_path "$SAVE" \
      --no_aggregate
    ;;
  ada*)
    B0=${ARM#ada}
    python -m eval.pipeline "baseline-fastdllm-t0.9" "$BASE_CONFIG" \
      --checkpoints self \
      --datasets gsm8k \
      --seeds "$SEED" \
      --temperatures 1.0 \
      --block_length "$B0" \
      --adaptive_block --delimiter_threshold "$DELIM_THRES" \
      "${FS_ARGS[@]}" \
      --save_path "$SAVE" \
      --no_aggregate
    ;;
  fixed64)
    # The constant bar. Unlike the ada arms, the block length appears nowhere in
    # the directory name -- only inside the generations filename and JSON, which
    # is what aggregate_results groups on. That is enough to keep b=64 apart from
    # anything else, but it means the directory alone does not tell you which
    # constant produced it.
    python -m eval.pipeline "baseline-fastdllm-t0.9" "$BASE_CONFIG" \
      --checkpoints self \
      --datasets gsm8k \
      --seeds "$SEED" \
      --temperatures 1.0 \
      --block_length 64 \
      "${FS_ARGS[@]}" \
      --save_path "$SAVE" \
      --no_aggregate
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 1
    ;;
esac

echo "=== done: arm=$ARM seed=$SEED shots=$SHOTS ==="

# Aggregate once the array has finished. Rows stay distinct because they differ
# in block_length ("cat" / "ada0.3_B16" / "ada0.3_B32" / "ada0.3_B64" / "64")
# and in few_shot, so the 0-shot and 5-shot halves of the table sit side by side:
#   python -m eval.aggregate_results \
#     --results_dir $WORK/eval_results/tau09_fs5 \
#     --results_dir $WORK/eval_results/adablock_b0_sweep \
#     --results_dir $WORK/eval_results/blocksweep \
#     --results_dir $WORK/eval_results/llada8b_blocksize_only \
#     --output_dir $WORK/eval_results/tau09_fs5
