#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=02:00:00
#SBATCH --job-name=bl32-dpls-alpha-int
#SBATCH --output=slurm/logs/%x_%j.out

# One 2-hour slice of the fixed-block (LUP, b=32, DPLS) alpha runs on ghx4-interactive,
# which starts in minutes when the ghx4 batch queue is days out. One node (4xGH200) per
# alpha, both alphas in parallel:
#   configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
#   -> $WORK/checkpoints/llada8b_bl32_a${A}_dpls, wandb run bl32-a${A}-dpls-v1
# A run is ~24 h (alpha=1 took 23.2 h), so ~14 slices; train.train auto-resumes from the
# last checkpoint and --save_steps 10 bounds what the time limit can lose to ~8 min.
# The interactive QoS allows one job per user, so slurm/logs/lup_bl32_alpha_chain.sh
# resubmits this file until both runs have written TRAIN_DONE.
#
#   ALPHAS="3 0p5" sbatch --export=ALL slurm/train_llada8b_bl32_dpls_alpha_int_deltaai.sbatch
#   ALPHAS=0p5 sbatch --export=ALL --nodes=1 slurm/train_llada8b_bl32_dpls_alpha_int_deltaai.sbatch
#
# RUNNING.lock is the one slurm/lib_cross_cluster_lock.sh uses, so the batch fallback
# (train_llada8b_bl32_dpls_alpha_deltaai.sbatch) never trains a run a slice is training,
# and a slice skips a run the batch job holds.

set -uo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
ALPHAS=${ALPHAS:-3 0p5}
LOCK_STALE_SEC=900

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16
cd "$REPO"
mkdir -p slurm/logs

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "=== slice $SLURM_JOB_ID on ${NODES[*]}, alphas: $ALPHAS, $(date)"

lock_is_busy() {  # $1 = lock file. Busy = fresh heartbeat from a holder that is alive.
  [ -f "$1" ] || return 1
  local age=$(( $(date +%s) - $(stat -c %Y "$1") ))
  [ "$age" -lt "$LOCK_STALE_SEC" ] || return 1
  # A slice killed by its time limit leaves a fresh lock behind; its job is gone.
  local holder
  holder=$(sed -n 's/^DeltaAI job \([0-9]*\) .*/\1/p' "$1")
  if [ -n "$holder" ] && [ -z "$(squeue -h -j "$holder" -t RUNNING -o %i 2>/dev/null)" ]; then
    return 1
  fi
  return 0
}

quarantine_broken_ckpt() {  # a kill in the middle of a save leaves a half-written dir
  local out=$1 last
  last=$(ls -d "$out"/checkpoint-[0-9]* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  [ -n "$last" ] || return 0
  local d=$out/checkpoint-$last
  if [ -f "$d/optimizer.pt" ] && [ -f "$d/scheduler.pt" ] && [ -f "$d/model.safetensors" ] \
     && python -c 'import json,sys; json.load(open(sys.argv[1]))' "$d/trainer_state.json" 2>/dev/null; then
    echo "[$(basename "$out")] resuming from checkpoint-$last"
  else
    echo "[$(basename "$out")] checkpoint-$last is incomplete; moving it aside"
    mv "$d" "$out/broken-checkpoint-$last-$SLURM_JOB_ID"
  fi
}

train_one() {  # $1 = alpha tag, $2 = node
  local A=$1 node=$2
  local out=$WORK/checkpoints/llada8b_bl32_a${A}_dpls
  local lock=$out/RUNNING.lock
  local log=slurm/logs/bl32-a${A}-dpls-int_${SLURM_JOB_ID}.out
  echo "DeltaAI job $SLURM_JOB_ID host $node since $(date -Is)" > "$lock"
  sleep 20
  if ! grep -q "job $SLURM_JOB_ID " "$lock"; then
    echo "[a$A] lost the lock race to: $(cat "$lock")"; return 0
  fi
  ( while true; do sleep 60; touch "$lock" 2>/dev/null || true; done ) &
  local hb=$!
  quarantine_broken_ckpt "$out"
  echo "[a$A] training on $node, log $log"
  # 4*32 = generation_batch_size 128, as in train_llada8b_bl32_a1_dpls_deltaai.sbatch.
  WANDB_RUN_ID=bl32-a${A}-dpls-v1 srun --nodes=1 --ntasks=1 --nodelist="$node" \
    --gpus-per-node=4 --cpus-per-task=64 \
    accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
    -m train.train \
    --config "configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml" \
    --per_device_train_batch_size 32 \
    --save_steps 10 \
    --output_dir "$out" > "$log" 2>&1
  local rc=$?
  kill "$hb" 2>/dev/null
  grep -q "job $SLURM_JOB_ID " "$lock" 2>/dev/null && rm -f "$lock"
  if [ "$rc" -eq 0 ]; then
    date > "$out/TRAIN_DONE"
    echo "[a$A] training complete; wrote $out/TRAIN_DONE"
  else
    echo "[a$A] train.train exited $rc (see $log)"
  fi
}

i=0
for A in $ALPHAS; do
  out=$WORK/checkpoints/llada8b_bl32_a${A}_dpls
  cfg=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
  [ -f "$cfg" ] || { echo "no config $cfg"; exit 2; }
  mkdir -p "$out"
  if [ -f "$out/TRAIN_DONE" ]; then echo "[a$A] TRAIN_DONE, skip"; continue; fi
  if lock_is_busy "$out/RUNNING.lock"; then
    echo "[a$A] busy: $(cat "$out/RUNNING.lock"), skip"; continue
  fi
  if [ "$i" -ge "${#NODES[@]}" ]; then echo "[a$A] no node left in this slice"; continue; fi
  train_one "$A" "${NODES[$i]}" &
  i=$((i + 1))
done
wait
echo "=== slice $SLURM_JOB_ID end $(date)"
