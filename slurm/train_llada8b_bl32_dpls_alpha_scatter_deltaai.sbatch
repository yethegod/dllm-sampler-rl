#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7000M
#SBATCH --time=02:00:00
#SBATCH --job-name=llada8b-bl32-dpls-alpha-sc
#SBATCH --output=slurm/logs/%x_%j.out

# Short-slice shape of the fixed-block alpha runs (see
# train_llada8b_bl32_dpls_alpha_deltaai.sbatch): 4 single-GPU srun tasks for 30 minutes.
# On a full ghx4 a 48-hour (or even 2-hour) node is a day away, while a 30-minute job
# backfills within the hour.
#
# The 4 GPUs may sit on one node or on several: with no whole node free, 1 + 1 + 2 spare
# GPUs on three nodes are still a slice. See the launcher at the bottom for how ranks are
# placed. (ghx4-interactive caps a job at 2 nodes; add --nodes=1 for the whole-node layout.)
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
EXTRA=(--save_steps 5)   # a 30-minute slice loses at most ~4 min to its time limit
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

# torchrun's contract, by hand, with ranks placed by GPU rather than by Slurm task.
# For a --gpus-per-task job Slurm lays out tasks (CPUs, memory) and GPUs independently:
# job 3678103 got 1 + 3 tasks but 2 + 2 GPUs on two nodes, and one srun over the tasks put
# two ranks on one GPU. So: read how many of the job's GPUs each node holds, then start
# one step per node with that many ranks. --gpu-bind=none lets every rank see all of its
# node's GPUs and take its own by LOCAL_RANK (accelerate: local rank modulo visible
# devices); with per-task binding NCCL fails ("invalid device ordinal") as soon as two
# ranks share a node.
TOTAL=$SLURM_NTASKS
# Per-node GPUs and CPUs straight from the job record, e.g.
#   Nodes=gh[093,152] CPU_IDs=0-7 Mem=56000 GRES=gpu:nvidia_gh200_120gb:2(IDX:0,3)
# (A probe step cannot tell: it inherits the job's one-GPU-per-task rule and sees only
# as many GPUs as it has tasks.)
NODES=(); CPN=(); declare -A GPUS
while read -r host g c; do
  [ -n "$host" ] || continue
  NODES+=("$host"); CPN+=("$c"); GPUS[$host]=$g
done < <(scontrol show job -d "$SLURM_JOB_ID" | python -c '
import re, subprocess, sys
for line in sys.stdin:
    m = re.match(r"\s+Nodes=(\S+) CPU_IDs=(\S+) Mem=\d+ GRES=(\S*)", line)
    if not m:
        continue
    hosts = subprocess.run(["scontrol", "show", "hostnames", m.group(1)],
                           capture_output=True, text=True).stdout.split()
    ncpu = 0
    for part in m.group(2).split(","):
        a, _, b = part.partition("-")
        ncpu += int(b or a) - int(a) + 1
    g = re.search(r":(\d+)\(IDX", m.group(3))
    for h in hosts:
        print(h, g.group(1) if g else 0, ncpu)
')
sum=0; ok=1; MASTER_ADDR=
for i in "${!NODES[@]}"; do
  n=${NODES[$i]}; g=${GPUS[$n]:-0}; c=${CPN[$i]:-0}
  echo "[layout] $n: $g GPU(s), $c CPU(s)"
  sum=$((sum + g))
  [ "$g" -eq 0 ] && continue
  [ -z "$MASTER_ADDR" ] && MASTER_ADDR=$n
  # 4 CPUs per rank is also the memory floor: 4 x --mem-per-cpu = 28 GB, a rank peaks at 21.
  [ $((c / g)) -ge 4 ] || ok=0
done
if [ "$sum" -ne "$TOTAL" ] || [ "$ok" -ne 1 ]; then
  echo "=== unusable layout on $SLURM_JOB_NODELIST ($sum of $TOTAL GPUs found, or a node with too few CPUs per GPU); not training in this slice."
  exit 3
fi
export MASTER_ADDR
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$TOTAL
export TRAIN_CONFIG=$CONFIG TRAIN_OUT=$OUTPUT_DIR
export TRAIN_EXTRA="${EXTRA[*]}"
echo "ranks: $WORLD_SIZE, rendezvous $MASTER_ADDR:$MASTER_PORT"
off=0; pids=()
for i in "${!NODES[@]}"; do
  n=${NODES[$i]}; g=${GPUS[$n]:-0}; c=${CPN[$i]}
  [ "$g" -eq 0 ] && continue
  RANK_OFFSET=$off srun --nodes=1 --ntasks="$g" --nodelist="$n" --gpus-per-task=1 --gpu-bind=none \
    --cpus-per-task=$((c / g)) --kill-on-bad-exit=1 bash -c '
      export RANK=$((RANK_OFFSET + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID
      export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
      echo "[rank $RANK] $(hostname -s) local rank $LOCAL_RANK CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} cpus $SLURM_CPUS_PER_TASK"
      exec python -m train.train --config "$TRAIN_CONFIG" --per_device_train_batch_size 32 \
        $TRAIN_EXTRA --output_dir "$TRAIN_OUT"' &
  pids+=($!)
  off=$((off + g))
done
# One step per node: if any of them dies, stop the others instead of letting them hang.
rc=0
for ((k = 0; k < ${#pids[@]}; k++)); do
  wait -n "${pids[@]}" || { rc=$?; kill "${pids[@]}" 2>/dev/null; break; }
done
wait
echo "=== train.train exited $rc, $(date), last ckpt: $(last_ckpt)"
if [ "$rc" -eq 0 ] && [ "$SMOKE" != 1 ]; then
  date > "$OUTPUT_DIR/TRAIN_DONE"
  echo "training complete; wrote $OUTPUT_DIR/TRAIN_DONE"
fi
exit "$rc"
