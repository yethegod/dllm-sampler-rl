#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=llada8b-bl32-dpls-alpha
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Alpha sweep of the fixed-block control (LUP: remasking=policy, dit_confidence, b=32,
# DPLS): train_llada8b_bl32_a1_dpls_deltaai.sbatch at alpha 0.5 or 3. A picks the config,
# OUTPUT_DIR and wandb run:
#   configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
#
# START:  sbatch -J llada8b-bl32-a0p5-dpls --export=ALL,A=0p5 slurm/train_llada8b_bl32_dpls_alpha_deltaai.sbatch
#         sbatch -J llada8b-bl32-a3-dpls   --export=ALL,A=3   slurm/train_llada8b_bl32_dpls_alpha_deltaai.sbatch
#
# The cluster is full, so the same run is queued in several shapes and whichever starts
# first trains; all of them honour $OUTPUT_DIR/RUNNING.lock (slurm/lib_cross_cluster_lock.sh):
#   - this file, 4xGH200 for 48h (the alpha=1 run took 23h in this shape);
#   - train_llada8b_bl32_dpls_alpha_2gpu_deltaai.sbatch, the same body on 2 GPUs
#     (per-device batch 64, ~half the step rate), which fits a partly used node;
#   - 2-hour slices on ghx4-interactive (train_llada8b_bl32_dpls_alpha_int_deltaai.sbatch).
# A job exits if TRAIN_DONE exists. If the lock is held by a weaker shape (an interactive
# slice, or a batch job with fewer GPUs) it scancels the holder and takes over from the
# last checkpoint; if the holder is an equal or stronger batch job it simply exits.
#
# STOP:   scancel -n <job name>

set -euo pipefail

A=${A:?set A=0p5 or A=3}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32_a${A}_dpls
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
NG=${SLURM_GPUS_ON_NODE:?}
# The 2-GPU wrapper sets SELF so that a re-queue keeps its shape.
SELF=${SELF:-$REPO/slurm/train_llada8b_bl32_dpls_alpha_deltaai.sbatch}
CLUSTER=DeltaAI

cd "$REPO"
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }
mkdir -p "$REPO/slurm/logs"
source slurm/lib_cross_cluster_lock.sh

preempt_weaker() {  # a continuous batch job beats a 2h slice; more GPUs beat fewer
  [ -f "$LOCK" ] || return 0
  local holder info hp hw st
  holder=$(sed -n 's/^DeltaAI job \([0-9]*\) .*/\1/p' "$LOCK")
  [ -n "$holder" ] || return 0
  info=$(squeue -h -j "$holder" -t RUNNING -o "%P %b" 2>/dev/null || true)
  # Not running, or squeue failed: leave it to the lock's own staleness rule.
  [ -n "$info" ] || return 0
  hp=${info%% *}; hw=${info##*:}
  [ "$hp" = ghx4-interactive ] && hw=0
  # A scattered slice (..._scatter_deltaai.sbatch, job name *-sc) runs at 4-GPU speed in
  # 2-hour pieces: a whole node beats it, the 2-GPU shape does not.
  case "$(squeue -h -j "$holder" -o %j 2>/dev/null)" in *-sc) hw=3 ;; esac
  case "$hw" in ''|*[!0-9]*) return 0 ;; esac
  if [ "$hw" -ge "$NG" ]; then
    # An equal or stronger batch job is already training this run and will finish it
    # (it re-queues itself at its time limit); polling from here would only add jobs.
    echo "job $holder ($info) already trains this run with >= $NG GPUs; exiting."
    exit 0
  fi
  echo "lock held by weaker job $holder ($info); cancelling it and taking over with $NG GPUs"
  scancel "$holder"
  for _ in $(seq 30); do
    sleep 10
    st=$(sacct -j "$holder" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
    case "$st" in CANCELLED*|COMPLETED|FAILED|TIMEOUT) rm -f "$LOCK"; return 0 ;; esac
  done
  echo "job $holder did not stop within 5 min; falling back to the normal lock rule"
}
preempt_weaker
take_lock_or_requeue

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bl32-a${A}-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

# A killed holder (scancel, time limit) can leave a half-written last checkpoint behind.
last=$(ls -d "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
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

# NG * per-device batch = generation_batch_size 128 (32 on 4 GPUs, 64 on 2).
echo "training alpha $A on $NG GPUs, per-device batch $((128 / NG))"
run_under_lock accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  --num_processes "$NG" \
  -m train.train \
  --config "$CONFIG" \
  --per_device_train_batch_size $((128 / NG)) \
  --save_steps 10 \
  --output_dir "$OUTPUT_DIR"
