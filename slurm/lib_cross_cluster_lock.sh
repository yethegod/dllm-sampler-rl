# Cross-cluster run lock for a single long training job queued on BOTH Delta and
# DeltaAI against one OUTPUT_DIR on the shared /work/hdd. Sourced by
# train_llada8b_block_unmask_cond_v2_dpls_alpha{,_deltaai}.sbatch.
#
# Whichever job starts first writes RUNNING.lock and touches it every 60 s. A job
# that finds a lock younger than LOCK_STALE_SEC does not train: it re-queues itself
# with --begin=now+1hour and exits, so the other cluster keeps polling until
# TRAIN_DONE appears or the holder dies (lock goes stale, the poller takes over and
# train.train resumes from the last checkpoint-*). A holder nearing its time limit
# (USR1, sent 5 min early by `#SBATCH --signal=B:USR1@300`) re-queues itself up to
# MAX_RESTARTS times; a plain TERM (scancel) only stops it. Simpler than the a0
# chunk chains: no chunking, no preemption.
#
# The caller sets OUTPUT_DIR, SELF (the sbatch file's repo path; Slurm runs a spool
# copy, so $0 is useless), CLUSTER and A (the alpha tag), then calls
# `take_lock_or_requeue` and `run_under_lock <command...>`.

DONE_MARKER=$OUTPUT_DIR/TRAIN_DONE
LOCK=$OUTPUT_DIR/RUNNING.lock
LOCK_STALE_SEC=900
RESTART=${RESTART:-0}
MAX_RESTARTS=${MAX_RESTARTS:-3}

requeue_self() {  # $1 = extra sbatch args, $2 = RESTART value for the new job
  sbatch $1 --job-name="$SLURM_JOB_NAME" \
    --export=ALL,A="$A",RESTART="$2",MAX_RESTARTS="$MAX_RESTARTS" "$SELF"
}

take_lock_or_requeue() {
  mkdir -p "$OUTPUT_DIR"
  if [ -f "$DONE_MARKER" ]; then
    echo "$DONE_MARKER exists; training already complete. Exiting."
    exit 0
  fi
  echo "[$CLUSTER restart $RESTART/$MAX_RESTARTS] job $SLURM_JOB_ID on $(hostname), $(date)"
  echo "latest checkpoint: $(ls -d "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | sed "s/.*checkpoint-//" | sort -n | tail -1 || echo none)"
  if [ -f "$LOCK" ]; then
    local age=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
    if [ "$age" -lt "$LOCK_STALE_SEC" ]; then
      echo "busy: $(cat "$LOCK") -- heartbeat ${age}s ago. Re-queueing myself in 1h, not training."
      requeue_self "--begin=now+1hour" "$RESTART"
      exit 0
    fi
    echo "stale lock (${age}s old): $(cat "$LOCK") -- taking over."
  fi
  echo "$CLUSTER job $SLURM_JOB_ID host $(hostname) since $(date -Is)" > "$LOCK"
  # Two clusters starting within the same minute both pass the check above; the
  # later write wins on Lustre, and the loser backs off here.
  sleep 30
  if ! grep -q "job $SLURM_JOB_ID " "$LOCK"; then
    echo "lost the lock race to: $(cat "$LOCK"). Re-queueing myself in 1h."
    requeue_self "--begin=now+1hour" "$RESTART"
    exit 0
  fi
  ( while true; do sleep 60; touch "$LOCK" 2>/dev/null || true; done ) &
  HEARTBEAT_PID=$!
  trap release_lock EXIT
}

release_lock() {
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  grep -q "job $SLURM_JOB_ID " "$LOCK" 2>/dev/null && rm -f "$LOCK"
}

run_under_lock() {
  # The command runs in the background so a signal is handled at once: forward TERM
  # to it, wait, clear the lock. USR1 (time limit ahead) also queues a successor.
  CHILD=
  stop_child() {
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null || true
    [ -n "$CHILD" ] && wait "$CHILD" 2>/dev/null || true
  }
  on_usr1() {
    stop_child
    if [ "$RESTART" -lt "$MAX_RESTARTS" ]; then
      echo "time limit ahead; queueing restart $((RESTART + 1))."
      requeue_self "" $((RESTART + 1))
    else
      echo "time limit ahead; MAX_RESTARTS=$MAX_RESTARTS reached, not re-queueing."
    fi
    release_lock; trap - EXIT; exit 143
  }
  on_term() { echo "TERM received (scancel?); stopping without re-queueing."; stop_child; release_lock; trap - EXIT; exit 143; }
  trap on_usr1 USR1
  trap on_term TERM INT
  "$@" &
  CHILD=$!
  local rc=0
  wait "$CHILD" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "command exited $rc; not marking done and not re-queueing (a crash, not a time limit)."
    exit "$rc"
  fi
  date > "$DONE_MARKER"
  echo "training complete; wrote $DONE_MARKER"
}
