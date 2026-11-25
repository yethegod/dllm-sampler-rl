#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=llada8b-bl32-dpls-alpha
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Delta (8xA100) twin of train_llada8b_bl32_dpls_alpha_deltaai.sbatch: the alpha sweep
# (A=0p5 or 3) of the fixed-block control (LUP: remasking=policy, dit_confidence, b=32,
# DPLS), same OUTPUT_DIR and wandb run. The config ships per_device_train_batch_size 16
# and 8*16 = 128, so no override.
#
# START:  sbatch -J llada8b-bl32-a3-dpls   --export=ALL,A=3   slurm/train_llada8b_bl32_dpls_alpha.sbatch
#         sbatch -J llada8b-bl32-a0p5-dpls --export=ALL,A=0p5 slurm/train_llada8b_bl32_dpls_alpha.sbatch
#
# DeltaAI trains the same runs in short slices (see the _deltaai and _scatter_deltaai
# files); /work/hdd is shared, so slurm/lib_cross_cluster_lock.sh keeps each run to one
# trainer: a job that finds a fresh RUNNING.lock re-queues itself for an hour, otherwise it
# resumes from the last checkpoint (the 4 -> 8 GPU change is handled by train.train).
# TRAIN_DONE written here is what DeltaAI's driver waits for before it evaluates.
#
# STOP:   scancel -n <job name>   (kills running + queued, on THIS cluster only)
# WATCH:  wandb run bl32-a${A}-dpls-v1; still_masked must be 0.

set -euo pipefail

A=${A:?set A=0p5 or A=3}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32_a${A}_dpls
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
SELF=$REPO/slurm/train_llada8b_bl32_dpls_alpha.sbatch
CLUSTER=Delta

cd "$REPO"
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }
mkdir -p "$REPO/slurm/logs"
source slurm/lib_cross_cluster_lock.sh
take_lock_or_requeue

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bl32-a${A}-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

# A DeltaAI slice killed by its time limit can leave a half-written last checkpoint.
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

run_under_lock accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config "$CONFIG" \
  --save_steps 10 \
  --output_dir "$OUTPUT_DIR"
