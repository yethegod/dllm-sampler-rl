#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=04:00:00
#SBATCH --job-name=llada8b-bu-a0-dpls-chunk
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sun01024@umn.edu

# The alpha=0 v2 joint run with DPLS unmask sampling (config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0_dpls.yaml),
# cut into 4h chunks, Delta side (8xA100). Same code path and chain machinery as the
# Bernoulli alpha=0 chain (train_llada8b_block_unmask_cond_v2a0_dpls_chunk.sbatch), with its
# own OUTPUT_DIR, lock and wandb run; the DeltaAI twin is
# train_llada8b_block_unmask_cond_v2a0_dpls_chunk_2gpu_deltaai.sbatch. The config ships
# per_device_train_batch_size 16 and 8*16 = generation_batch_size 128, so no override.
# Pass a longer limit with `sbatch --time=08:00:00 ...` if the gpuA100x8 queue takes
# 8h jobs readily.
#
# START:  sbatch slurm/train_llada8b_block_unmask_cond_v2a0_dpls_chunk.sbatch

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
#      queued on both (train_llada8b_block_unmask_cond_v2a0_dpls_chunk.sbatch on Delta,
#      the _deltaai one here). Whichever chunk starts first writes RUNNING.lock
#      and touches it every 60 s. A chunk that finds a lock younger than 15 min
#      does NOT train and does NOT queue an afterany successor; it re-queues
#      itself with --begin=now+1hour at the same CHUNK_INDEX and exits, so the
#      other cluster's chain keeps polling cheaply until TRAIN_DONE appears or
#      the running chain dies (lock goes stale, poller takes over). The lock is
#      removed on any exit, including SIGTERM at the time limit.
#
# STOP:   scancel -n <job name>   (kills running + queued, on THIS cluster only)
# WATCH:  wandb run bu-a0-dpls-v1 (block_size/frac_*, block_entropy, tokens_per_step_mean;
#         still_masked must stay 0)
#
# Same watch-list as the a1 run: block_size/frac_* and block_entropy for the
# block head, mean_unmask_prob -> 0 or NaN loss as the kill signals. A slow
# policy (num_steps_mean > 100) is the expected shape at alpha=0, not a failure.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a0_dpls
DONE_MARKER=$OUTPUT_DIR/TRAIN_DONE
LOCK=$OUTPUT_DIR/RUNNING.lock
LOCK_STALE_SEC=900
YIELD=$OUTPUT_DIR/YIELD
# GPUs this chunk holds. A chunk arriving with MORE GPUs than the current lock
# holder preempts it (see the lock block); equal or fewer backs off.
WEIGHT=8
YIELD_WAIT_SEC=900
# A writer outside this protocol (e.g. a Delta job launched by hand, like wandb run
# u1nji1hh on 2026-09-23) holds no lock but keeps saving checkpoints. Every chunk that
# held the lock stamps LOCK_RELEASED on its way out, so a checkpoint NEWER than that
# stamp was written by someone else; if it is also younger than FOREIGN_WRITER_SEC the
# directory is treated as busy. 1h covers the slowest save cadence (25 steps x ~80 s).
RELEASED=$OUTPUT_DIR/LOCK_RELEASED
FOREIGN_WRITER_SEC=3600
SELF=$REPO/slurm/train_llada8b_block_unmask_cond_v2a0_dpls_chunk.sbatch

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

NEWEST=$(ls -td "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | head -1 || true)
if [ ! -f "$LOCK" ] && [ -n "$NEWEST" ]; then
  NEWEST_T=$(stat -c %Y "$NEWEST")
  REL_T=0; [ -f "$RELEASED" ] && REL_T=$(stat -c %Y "$RELEASED")
  NEWEST_AGE=$(( $(date +%s) - NEWEST_T ))
  if [ "$NEWEST_T" -gt "$REL_T" ] && [ "$NEWEST_AGE" -lt "$FOREIGN_WRITER_SEC" ]; then
    echo "busy: $(basename "$NEWEST") was saved ${NEWEST_AGE}s ago by a job that holds no lock. Re-queueing myself in 1h, not training."
    sbatch --begin=now+1hour --export=ALL,CHUNK_INDEX="$CHUNK_INDEX",MAX_CHUNKS="$MAX_CHUNKS" "$SELF"
    exit 0
  fi
fi
if [ -f "$LOCK" ]; then
  LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
  if [ "$LOCK_AGE" -lt "$LOCK_STALE_SEC" ]; then
    HOLDER_WEIGHT=$(sed -n 's/.*weight=\([0-9]*\).*/\1/p' "$LOCK" | head -1)
    HOLDER_WEIGHT=${HOLDER_WEIGHT:-99}
    if [ "$WEIGHT" -gt "$HOLDER_WEIGHT" ]; then
      # A smaller job holds the run. Ask it to stop (its heartbeat loop watches
      # YIELD), then wait for the lock to clear. Progress it loses is bounded by
      # its save_steps; this slot took days to get and is worth more.
      echo "preempting: $(cat "$LOCK") (weight $HOLDER_WEIGHT < mine $WEIGHT). Writing YIELD and waiting up to ${YIELD_WAIT_SEC}s."
      echo "weight=$WEIGHT job $SLURM_JOB_ID host $(hostname) $(date -Is)" > "$YIELD"
      WAITED=0
      while [ -f "$LOCK" ] && [ "$WAITED" -lt "$YIELD_WAIT_SEC" ]; do sleep 15; WAITED=$((WAITED + 15)); done
      rm -f "$YIELD"
      if [ -f "$LOCK" ]; then
        LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
        if [ "$LOCK_AGE" -lt "$LOCK_STALE_SEC" ]; then
          echo "holder did not yield within ${YIELD_WAIT_SEC}s; re-queueing myself in 1h."
          sbatch --begin=now+1hour --export=ALL,CHUNK_INDEX="$CHUNK_INDEX",MAX_CHUNKS="$MAX_CHUNKS" "$SELF"
          exit 0
        fi
        echo "holder's lock went stale while waiting; taking over."
      else
        echo "holder yielded after ${WAITED}s; taking over."
      fi
    else
      echo "busy: $(cat "$LOCK") -- heartbeat ${LOCK_AGE}s ago, weight $HOLDER_WEIGHT >= mine $WEIGHT. Re-queueing myself in 1h, not training."
      sbatch --begin=now+1hour --export=ALL,CHUNK_INDEX="$CHUNK_INDEX",MAX_CHUNKS="$MAX_CHUNKS" "$SELF"
      exit 0
    fi
  fi
  [ -f "$LOCK" ] && echo "stale lock (${LOCK_AGE}s old): $(cat "$LOCK") -- taking over."
fi
rm -f "$YIELD"   # a stale request from an earlier arrival must not stop us
echo "weight=$WEIGHT Delta job $SLURM_JOB_ID host $(hostname) since $(date -Is)" > "$LOCK"
# Heartbeat, and the yield watch: a bigger job wrote YIELD -> TERM ourselves, which
# runs on_term (stop accelerate, clear the lock, exit 143). The afterany
# successor is already queued and will find the bigger job's lock.
( while true; do sleep 60; touch "$LOCK" 2>/dev/null || true;
    if [ -f "$YIELD" ]; then
      YW=$(sed -n 's/.*weight=\([0-9]*\).*/\1/p' "$YIELD" | head -1); YW=${YW:-0}
      if [ "$YW" -gt "$WEIGHT" ]; then echo "[chunk $CHUNK_INDEX] YIELD from $(cat "$YIELD"); stopping."; kill -TERM $$; break; fi
    fi
  done ) &
HEARTBEAT_PID=$!
cleanup() { kill "$HEARTBEAT_PID" 2>/dev/null || true; rm -f "$LOCK"; date > "$RELEASED"; }
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
# One wandb run for the whole chain instead of one per chunk. Override with
# WANDB_RUN_ID=... at submit time.
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a0-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16
mkdir -p "$HF_HOME"

# --save_steps 25 overrides the config's 50 so a timed-out chunk loses less.
accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0_dpls.yaml \
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
