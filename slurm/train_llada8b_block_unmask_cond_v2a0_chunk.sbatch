#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=04:00:00
#SBATCH --job-name=llada8b-block-unmask-cond-v2a0-chunk
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sun01024@umn.edu

# The alpha=0 v2 joint run (config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0.yaml)
# cut into 4h chunks, Delta side (8xA100). Same run, same OUTPUT_DIR as the
# DeltaAI chain (train_llada8b_block_unmask_cond_v2a0_chunk_deltaai.sbatch): the
# config ships per_device_train_batch_size 16, and 8*16 = generation_batch_size
# 128, so no batch-size override here. Memory is well inside 80 GB: the GH200
# side runs 32 per device with the same timestep_batch_size 32 chunking.
# A100s are slower per step than GH200s; the two chains just hand the same
# checkpoints back and forth via the lock below, so pace only changes how many
# chunks the run needs (MAX_CHUNKS). Pass a longer limit with
# `sbatch --time=08:00:00 ...` if the gpuA100x8 queue takes 8h jobs readily.
#
# START:  sbatch slurm/train_llada8b_block_unmask_cond_v2a0_chunk.sbatch

# HOW THE CHAIN WORKS
#   1. At start, this job submits one more copy of itself with
#      --dependency=afterany:<this job>, so the next chunk is queued (and starts
#      accruing priority) while this one runs. afterany, not afterok: a chunk that
#      hits its time limit exits non-zero and must still be followed.
#   2. train.train auto-resumes from the latest checkpoint-* in OUTPUT_DIR; with
#      --save_steps 25 (15 MB each) a chunk loses at most ~25 steps on timeout.
#   3. When accelerate exits 0 the epoch is complete: TRAIN_DONE is written and
#      every later chunk exits at once, so the chain ends by itself.
#   4. CHUNK_INDEX (exported to each child) caps the chain at MAX_CHUNKS so a
#      persistent crash cannot resubmit forever. Raise MAX_CHUNKS if the run is
#      slower than expected and legitimately needs more chunks.
#   5. CROSS-CLUSTER LOCK. Delta and DeltaAI share /work/hdd, and a chain may be
#      queued on both (train_llada8b_block_unmask_cond_v2a0_chunk.sbatch on Delta,
#      the _deltaai one here). Whichever chunk starts first writes RUNNING.lock
#      and touches it every 60 s. A chunk that finds a lock younger than 15 min
#      does NOT train and does NOT queue an afterany successor; it re-queues
#      itself with --begin=now+1hour at the same CHUNK_INDEX and exits, so the
#      other cluster's chain keeps polling cheaply until TRAIN_DONE appears or
#      the running chain dies (lock goes stale, poller takes over). The lock is
#      removed on any exit, including SIGTERM at the time limit.
#
# STOP:   scancel -n <job name>   (kills running + queued, on THIS cluster only)
# WATCH:  python -m eval.monitor_block_head $WORK/checkpoints/llada8b_block_unmask_cond_v2_a0
#
# Same watch-list as the a1 run: block_size/frac_* and block_entropy for the
# block head, mean_unmask_prob -> 0 or NaN loss as the kill signals. A slow
# policy (num_steps_mean > 100) is the expected shape at alpha=0, not a failure.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a0
DONE_MARKER=$OUTPUT_DIR/TRAIN_DONE
LOCK=$OUTPUT_DIR/RUNNING.lock
LOCK_STALE_SEC=900
SELF=$REPO/slurm/train_llada8b_block_unmask_cond_v2a0_chunk.sbatch

CHUNK_INDEX=${CHUNK_INDEX:-1}
MAX_CHUNKS=${MAX_CHUNKS:-14}   # 14 x 4h = 56h of wall clock, ~2x the a1 pace

cd "$REPO"
mkdir -p "$OUTPUT_DIR" "$REPO/slurm/logs"

if [ -f "$DONE_MARKER" ]; then
  echo "[chunk $CHUNK_INDEX] $DONE_MARKER exists; training already complete. Exiting."
  exit 0
fi

echo "[chunk $CHUNK_INDEX/$MAX_CHUNKS] Delta job $SLURM_JOB_ID on $(hostname), $(date)"
echo "latest checkpoint: $(ls -d "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | sed "s/.*checkpoint-//" | sort -n | tail -1 || echo none)"

if [ -f "$LOCK" ]; then
  LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
  if [ "$LOCK_AGE" -lt "$LOCK_STALE_SEC" ]; then
    echo "busy: $(cat "$LOCK") -- heartbeat ${LOCK_AGE}s ago. Re-queueing myself in 1h, not training."
    sbatch --begin=now+1hour --export=ALL,CHUNK_INDEX="$CHUNK_INDEX",MAX_CHUNKS="$MAX_CHUNKS" "$SELF"
    exit 0
  fi
  echo "stale lock (${LOCK_AGE}s old): $(cat "$LOCK") -- taking over."
fi
echo "Delta job $SLURM_JOB_ID host $(hostname) since $(date -Is)" > "$LOCK"
( while true; do sleep 60; touch "$LOCK" 2>/dev/null || true; done ) &
HEARTBEAT_PID=$!
cleanup() { kill "$HEARTBEAT_PID" 2>/dev/null || true; rm -f "$LOCK"; }
# EXIT covers normal completion and set -e failures; TERM/INT (Slurm time limit,
# scancel) must also stop the script so the DONE marker below is never written.
trap cleanup EXIT
# accelerate runs in the background and the script waits on it, so a TERM (Slurm
# time limit, scancel) is handled at once instead of after Slurm's SIGKILL: forward
# it to the launcher, wait for it, clear the lock, exit. Chunk 1 (job 3177775) ran
# accelerate in the foreground and left RUNNING.lock behind on TIMEOUT.
CHILD=
on_term() { [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true; [ -n "$CHILD" ] && wait "$CHILD" 2>/dev/null || true; cleanup; trap - EXIT; exit 143; }
trap on_term TERM INT

# Queue the next chunk before doing anything slow.
if [ "$CHUNK_INDEX" -lt "$MAX_CHUNKS" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,CHUNK_INDEX=$((CHUNK_INDEX + 1)),MAX_CHUNKS="$MAX_CHUNKS" "$SELF")
  echo "queued chunk $((CHUNK_INDEX + 1)) as job $NEXT (afterany:$SLURM_JOB_ID)"
else
  echo "MAX_CHUNKS=$MAX_CHUNKS reached; not queueing a successor."
fi

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16
mkdir -p "$HF_HOME"

# --save_steps 25 overrides the config's 50 so a timed-out chunk loses less.
accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0.yaml \
  --save_steps 25 \
  --output_dir "$OUTPUT_DIR" &
CHILD=$!
RC=0
wait "$CHILD" || RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[chunk $CHUNK_INDEX] accelerate exited $RC (time limit or error); not marking done."
  exit "$RC"
fi

# Only reached if accelerate exited 0, i.e. trainer.train() returned: epoch done.
date > "$DONE_MARKER"
echo "[chunk $CHUNK_INDEX] training complete; wrote $DONE_MARKER"
