#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=08:00:00
#SBATCH --job-name=llada8b-block-unmask-cond-v2a0-a100x4
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sun01024@umn.edu

# The alpha=0 v2 joint run (config header:
# configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0.yaml)
# in 8h chunks on Delta 4xA100-40GB (gpuA100x4). Same OUTPUT_DIR and lock as the
# other chains; 4*32 = generation_batch_size 128.
#
# MEMORY. No run of this path at per-device 32 on an A100 exists. The only 4xA100
# run in wandb (z06p8wn9, 2026-07-13) died at step 0 on the batch invariant
# (16 x 4 = 64 != 128), not on memory. The a0 run measured 57.9 GiB (NVML, so it
# includes allocator slack) at per-device 64 on GH200; peak ~ 16.0 GiB weights +
# 0.655 GiB/sequence puts per-device 32 near 37 GiB, ~3 GiB under the card. Two
# settings that leave the run unchanged buy margin: --timestep_batch_size 16 (yaml:
# 32) halves the compute_loss time slice, and compute_loss backwards each slice
# under no_sync, so the gradient is the same; expandable_segments cuts allocator
# fragmentation. Watch the first ~15 min (first rollout + loss). If it OOMs, use
# the 8xA100 script instead (per-device 16, ~27 GiB).
#
# PRE-FLIGHT. Before taking the lock or queueing a successor, the chunk checks that
# the env can import the trainer. On 2026-09-23 two DeltaAI chunks each waited a
# day in the queue and then died in 40 s on an ImportError after a CadLLM
# requirements install downgraded transformers; this chain stops loudly instead.
#
# START:  sbatch slurm/train_llada8b_block_unmask_cond_v2a0_chunk_a100x4.sbatch
# STOP:   scancel -n llada8b-block-unmask-cond-v2a0-a100x4

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
YIELD=$OUTPUT_DIR/YIELD
# Speed rank, not GPU count: a chunk arriving with a HIGHER weight than the lock
# holder preempts it; equal or lower backs off. 4xA100-40GB is estimated ~1.3-1.5x
# SLOWER per step than DeltaAI 2xGH200 (weight 2; 107-129 s/step measured), so it
# gets the same weight: neither preempts the other. 4xGH200 (4) and 8xA100 (8)
# outrank both.
WEIGHT=2
YIELD_WAIT_SEC=900
SELF=$REPO/slurm/train_llada8b_block_unmask_cond_v2a0_chunk_a100x4.sbatch

CHUNK_INDEX=${CHUNK_INDEX:-1}
MAX_CHUNKS=${MAX_CHUNKS:-14}   # 14 x 8h; ~1450 steps left at an estimated ~150 s/step is ~60h

cd "$REPO"
mkdir -p "$OUTPUT_DIR" "$REPO/slurm/logs"

if [ -f "$DONE_MARKER" ]; then
  echo "[chunk $CHUNK_INDEX] $DONE_MARKER exists; training already complete. Exiting."
  exit 0
fi

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm
if ! PF_ERR=$(python -c "import trl.scripts.utils, train.trainer" 2>&1 >/dev/null); then
  echo "[chunk $CHUNK_INDEX] PRE-FLIGHT FAILED: the dllm env cannot import the trainer."
  echo "Not training, not taking the lock, not queueing a successor. Last lines:"
  echo "$PF_ERR" | tail -n 5
  exit 1
fi

echo "[chunk $CHUNK_INDEX/$MAX_CHUNKS] Delta job $SLURM_JOB_ID on $(hostname), $(date)"
echo "latest checkpoint: $(ls -d "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | sed "s/.*checkpoint-//" | sort -n | tail -1 || echo none)"

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

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

# --per_device_train_batch_size 32 is mandatory on 4 GPUs (4*32 = 128; trainer asserts it).
# --timestep_batch_size 16: memory margin, gradient-equivalent (see MEMORY above).
# --save_steps 10: a faster chain (4xGH200, 8xA100) may preempt this one.
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0.yaml \
  --per_device_train_batch_size 32 \
  --timestep_batch_size 16 \
  --save_steps 10 \
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
