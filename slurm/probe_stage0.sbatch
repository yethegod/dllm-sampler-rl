#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --job-name=probe-stage0
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Stage 0 go/no-go: is the observation the bottleneck at all?
#
# Every policy here sees one number per position (top-1 confidence). The hidden-state
# projector policies exist on the bet that the dLLM's 4096-d state separates good
# unmasking decisions better. This job tests that bet for ~1 GPU-hour instead of the
# multi-day run it would otherwise take to find out.
#
# Decoding is a FIXED baseline, never a policy under training, so the label cannot
# depend on the thing being evaluated. The label is free: x0 is already computed every
# step, and comparing it to the final sequence gives, for each candidate position,
# "would unmasking here have committed the token that eventually won?".
#
# ROWS ARE ACTIONABLE POSITIONS ONLY (probe schema 2). The earlier runs scored every
# masked position, which is dominated by the not-yet-reached tail of the sequence --
# positions no decoding rule was allowed to pick, whose x0 is an unconditioned guess.
# Those easy negatives lift every probe's AUC together and answer a question the policy
# is never asked. This version restricts to positions that were masked AND inside the
# block being decoded at that step. Schema 1 files are refused by eval.fit_probes rather
# than silently pooled, so the two generations of data cannot be compared by accident.
#
# Two arms, ARM=fastdllm (default) or ARM=random:
#   fastdllm   Fast-dLLM at thres 0.9 -- the parallel baseline the repo actually runs.
#   random     256 steps, one position per step -- the sequential control. Its
#              actionable set is the same block-local one, but the trajectory is not
#              confidence-ordered, so a probe that only works on Fast-dLLM trajectories
#              is measuring the baseline's own ordering rather than the observation.
# Both arms share --seed so their example splits and subsampling are comparable.
#
# What the three AUCs mean:
#   hidden ~= top-1    the observation is not the bottleneck. Abandon the hidden-input
#                      line and look at reward / credit assignment instead -- which is
#                      also what the block-policy adaptivity diagnosis pointed at.
#   top-16 ~= hidden   set confidences_top_p: 16 (one line, +1920 params) and skip the
#                      projector, the replay path, and the retuning that comes with them.
#   hidden >  both     the bet is live, and a dense-label warm-start has signal to teach.
#
# Sizing, corrected after the first run (jobs 21449052 / 21449753). That run collected
# 438k rows from only 88 examples and the 4096-d probe overfit at the example level:
# rows inside one generation share a trajectory, so the effective sample size is the
# number of EXAMPLES, not rows. This one trades rows for examples -- 384 examples,
# replaying every 4th step -- which costs about the same GPU time because the replay
# pass shrinks by the same factor. Fast-dLLM at 0.9 takes T=205-255 steps at
# block_length 32 (not the 81 recorded at block_length 128), so a batch is roughly
# T + T/4 forwards. The random arm is a flat T=256.
#
# EVERY RUN RECOLLECTS. There is no reuse switch: the row support changed underneath
# these filenames once already, and refitting a stale file is the exact failure the
# schema exists to prevent -- a flag that skips collection is a flag that will
# eventually be set by accident. An existing $OUT is not lost either; the collector
# moves it to a timestamped .bak.npz sibling before writing.
#
# Usage:
#   sbatch slurm/probe_stage0.sbatch                # fastdllm arm
#   ARM=random sbatch slurm/probe_stage0.sbatch     # random-256 arm

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9

ARM=${ARM:-fastdllm}
SEED=${SEED:-42}
N_TEST=${N_TEST:-384}
BATCH_SIZE=${BATCH_SIZE:-16}
STEP_STRIDE=${STEP_STRIDE:-4}
ROWS_PER_STEP=${ROWS_PER_STEP:-96}
DATASET=${DATASET:-gsm8k}

case "$ARM" in
  fastdllm)
    NAME=${TAG:-gsm8k_t09_actionable_fastdllm}
    ARM_ARGS=(--remasking fastdllm --thres "${THRES:-0.9}")
    ;;
  random)
    NAME=${TAG:-gsm8k_random_k256_actionable}
    ARM_ARGS=(--remasking random --steps "${STEPS:-256}")
    ;;
  *)
    echo "unknown ARM=$ARM (expected 'fastdllm' or 'random')" >&2
    exit 2
    ;;
esac
OUT=$WORK/probe_data/${NAME}.npz

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$WORK/probe_data" "$REPO/slurm/logs"
cd "$REPO"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== commit ==="
# Full sha, not %h: the NPZ records the full one, and this log is what ties a result
# back to it. The diff status is the other half -- a clean sha reproduces, a dirty one
# does not, and only saying so up front makes that recoverable later.
git -C "$REPO" rev-parse HEAD
git -C "$REPO" log -1 --format='%s'
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "tracked working tree: DIRTY"
  git -C "$REPO" status --porcelain --untracked-files=no
  git -C "$REPO" diff --stat
else
  echo "tracked working tree: clean"
fi
echo "=== arm ==="
echo "ARM=$ARM  OUT=$OUT  SEED=$SEED  N_TEST=$N_TEST  BATCH_SIZE=$BATCH_SIZE"
echo "STEP_STRIDE=$STEP_STRIDE  ROWS_PER_STEP=$ROWS_PER_STEP  ${ARM_ARGS[*]}"

echo
echo "############ collect ############"
# Always fresh. The collector backs up any existing $OUT to a timestamped .bak.npz
# before it writes, so this never destroys an earlier result.
python -u -m eval.collect_probe_data \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --dataset "$DATASET" \
  --n_test "$N_TEST" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  "${ARM_ARGS[@]}" \
  --step_stride "$STEP_STRIDE" \
  --rows_per_step "$ROWS_PER_STEP" \
  --out "$OUT"

echo
echo "############ fit probes ############"
# Split seed shared across arms so the fastdllm and random numbers differ because of
# the trajectories, not because of which examples landed in test.
python -u -m eval.fit_probes \
  --data "$OUT" --epochs "${EPOCHS:-400}" --seed "${SPLIT_SEED:-0}"
