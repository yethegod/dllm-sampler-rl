#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4500M
#SBATCH --time=02:00:00
#SBATCH --job-name=llada8b-bl32-dpls-alpha-sc
#SBATCH --output=slurm/logs/%x_%j.out

# "Scattered" shape of the fixed-block alpha runs (see
# train_llada8b_bl32_dpls_alpha_deltaai.sbatch): 4 single-GPU tasks wherever the cluster
# has a free GPU, usually on 4 different nodes, for 2 hours. On a full ghx4 a whole node
# (or two free GPUs on one node) is a day away, while single GPUs with <= ~36 GB of host
# memory free up all the time and a 2-hour job backfills into them.
#
# The run itself is unchanged: 4 DDP ranks x per-device batch 32 = generation_batch_size
# 128. Only the launcher differs: `accelerate launch` spawns its ranks on one node, so each
# rank is an srun task here and gets RANK / WORLD_SIZE / MASTER_ADDR the way torchrun
# would set them; accelerate then builds the same NCCL process group across nodes.
#
# One job = one slice; the shorter its time limit, the sooner it backfills (a 25-minute
# 4-GPU job started within the hour when 2-hour ones were a day out). Lock:
# $OUTPUT_DIR/RUNNING.lock, shared with the other shapes; a whole-node batch job that
# starts cancels a slice and resumes its checkpoints.
#
# Batch (ghx4), one self-renewing chain per alpha: a slice queues its successor
# (afterany) before training, so the chain lives until TRAIN_DONE; it stops by itself
# after two slices in a row that add no checkpoint. If another job holds the lock the
# slice re-queues itself for 20 minutes later and exits at once.
#   sbatch -J llada8b-bl32-a3-sc   --time=01:00:00 --export=ALL,A=3   <this file>
#   sbatch -J llada8b-bl32-a0p5-sc --time=01:00:00 --export=ALL,A=0p5 <this file>
# Interactive (ghx4-interactive, one job per user, so no successor: NOCHAIN=1 and the
# login-node driver slurm/logs/lup_bl32_alpha_chain2.sh resubmits). A=auto trains
# whichever unfinished alpha nobody holds, the one with fewer checkpoints first.
#   sbatch -p ghx4-interactive -J bl32-sc-int --time=01:00:00 --export=ALL,A=auto,NOCHAIN=1 <this file>
# SMOKE:  sbatch -J bl32-sc-smoke --time=00:25:00 --export=ALL,A=3,SMOKE=1 <this file>
#         3 optimizer steps into smoke_bl32_scatter, no lock, no successor.
# STOP:   scancel -n <job name>

set -uo pipefail

A=${A:?set A=0p5, A=3 or A=auto}
SMOKE=${SMOKE:-0}
NOCHAIN=${NOCHAIN:-0}
SLICE=${SLICE:-1}
STALL=${STALL:-0}
PREV_CKPT=${PREV_CKPT:-}
MAX_SLICES=${MAX_SLICES:-60}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9

lock_busy() {  # $1 = lock file. Busy = fresh heartbeat from a job that is still running.
  [ -f "$1" ] || return 1
  local age=$(( $(date +%s) - $(stat -c %Y "$1") )) holder
  [ "$age" -lt 900 ] || return 1
  holder=$(sed -n 's/^DeltaAI job \([0-9]*\) .*/\1/p' "$1")
  [ -z "$holder" ] || [ -n "$(squeue -h -j "$holder" -t RUNNING -o %i 2>/dev/null)" ]
}
if [ "$A" = auto ]; then
  best=; bestn=
  for cand in 3 0p5; do
    o=$WORK/checkpoints/llada8b_bl32_a${cand}_dpls
    [ -f "$o/TRAIN_DONE" ] && continue
    lock_busy "$o/RUNNING.lock" && continue
    n=$(ls -d "$o"/checkpoint-[0-9]* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [ -z "$best" ] || [ "${n:-0}" -lt "$bestn" ]; then best=$cand; bestn=${n:-0}; fi
  done
  [ -n "$best" ] || { echo "auto: every unfinished alpha is held by another job (or all are done); nothing to do."; exit 0; }
  A=$best
  echo "auto: picked alpha $A (last checkpoint ${bestn})"
fi
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32_a${A}_dpls
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
SELF=$REPO/slurm/train_llada8b_bl32_dpls_alpha_scatter_deltaai.sbatch
EXTRA=(--save_steps 10)
if [ "$SMOKE" = 1 ]; then
  OUTPUT_DIR=$WORK/checkpoints/smoke_bl32_scatter
  EXTRA=(--max_steps 3 --save_steps 2 --warmup_steps 2)
  export WANDB_MODE=offline
fi
LOCK=$OUTPUT_DIR/RUNNING.lock

cd "$REPO"
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 2; }
mkdir -p "$OUTPUT_DIR" slurm/logs
last_ckpt() { ls -d "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
echo "=== slice $SLICE (stall $STALL) alpha $A job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, $(date), last ckpt: $(last_ckpt)"

if [ "$SMOKE" != 1 ]; then
  if [ -f "$OUTPUT_DIR/TRAIN_DONE" ]; then echo "TRAIN_DONE exists; exiting."; exit 0; fi
  NOW_CKPT=$(last_ckpt)
  # The successor keeps this job's time limit (command-line --time is not in $SELF).
  TL=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null); TL=${TL:-01:00:00}
  if lock_busy "$LOCK"; then
    echo "busy: $(cat "$LOCK"); this slice does nothing."
    if [ "$NOCHAIN" != 1 ]; then
      sbatch --parsable --begin=now+20minutes --time="$TL" --job-name="$SLURM_JOB_NAME" \
        --export=ALL,A="$A",SLICE="$SLICE",STALL=0,PREV_CKPT="$NOW_CKPT",MAX_SLICES="$MAX_SLICES" "$SELF" \
        | sed 's/^/re-queued for 20 min later as job /'
    fi
    exit 0
  fi
  if [ "$SLICE" -gt 1 ] && [ "$NOW_CKPT" = "$PREV_CKPT" ]; then STALL=$((STALL + 1)); else STALL=0; fi
  if [ "$STALL" -ge 2 ]; then echo "two slices without a new checkpoint; chain stops here."; exit 1; fi
  if [ "$NOCHAIN" != 1 ] && [ "$SLICE" -lt "$MAX_SLICES" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" --time="$TL" --job-name="$SLURM_JOB_NAME" \
      --export=ALL,A="$A",SLICE=$((SLICE + 1)),STALL="$STALL",PREV_CKPT="$NOW_CKPT",MAX_SLICES="$MAX_SLICES" "$SELF") \
      && echo "queued slice $((SLICE + 1)) as job $NEXT"
  fi
  echo "DeltaAI job $SLURM_JOB_ID host $(hostname) since $(date -Is)" > "$LOCK"
  sleep 20
  grep -q "job $SLURM_JOB_ID " "$LOCK" || { echo "lost the lock race to: $(cat "$LOCK")"; exit 0; }
  ( while true; do sleep 60; touch "$LOCK" 2>/dev/null || true; done ) &
  HB=$!
  trap 'kill $HB 2>/dev/null; grep -q "job $SLURM_JOB_ID " "$LOCK" 2>/dev/null && rm -f "$LOCK"' EXIT
else
  rm -rf "$OUTPUT_DIR"; mkdir -p "$OUTPUT_DIR"
fi

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bl32-a${A}-dpls-v1}
[ "$SMOKE" = 1 ] && export WANDB_RUN_ID=bl32-scatter-smoke-$SLURM_JOB_ID
export WANDB_RESUME=allow
export OMP_NUM_THREADS=8

# A killed slice can leave a half-written last checkpoint behind.
last=$(last_ckpt)
if [ -n "$last" ]; then
  d=$OUTPUT_DIR/checkpoint-$last
  if [ -f "$d/optimizer.pt" ] && [ -f "$d/scheduler.pt" ] && [ -f "$d/model.safetensors" ] \
     && python -c 'import json,sys; json.load(open(sys.argv[1]))' "$d/trainer_state.json" 2>/dev/null; then
    echo "resuming from checkpoint-$last"
  else
    echo "checkpoint-$last is incomplete; moving it aside"
    mv "$d" "$OUTPUT_DIR/broken-checkpoint-$last-$SLURM_JOB_ID"
  fi
fi

# torchrun's contract, by hand: rank 0 runs on the first node of the allocation (block
# distribution) and hosts the rendezvous. --gpu-bind=none: every task sees all of the
# job's GPUs on its node and picks its own by LOCAL_RANK (accelerate: local rank modulo
# visible device count). With the default per-task binding each task sees only "GPU 0",
# and NCCL fails with "invalid device ordinal" as soon as two ranks share a node.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS
export TRAIN_CONFIG=$CONFIG TRAIN_OUT=$OUTPUT_DIR
export TRAIN_EXTRA="${EXTRA[*]}"
echo "ranks: $WORLD_SIZE, rendezvous $MASTER_ADDR:$MASTER_PORT"
srun --ntasks="$SLURM_NTASKS" --gpus-per-task=1 --gpu-bind=none --cpus-per-task=8 \
  --distribution=block --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
    echo "[rank $RANK] $(hostname) local rank $LOCAL_RANK CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    exec python -m train.train --config "$TRAIN_CONFIG" --per_device_train_batch_size 32 \
      $TRAIN_EXTRA --output_dir "$TRAIN_OUT"'
rc=$?
echo "=== train.train exited $rc, $(date), last ckpt: $(last_ckpt)"
if [ "$rc" -eq 0 ] && [ "$SMOKE" != 1 ]; then
  date > "$OUTPUT_DIR/TRAIN_DONE"
  echo "training complete; wrote $OUTPUT_DIR/TRAIN_DONE"
fi
exit "$rc"
