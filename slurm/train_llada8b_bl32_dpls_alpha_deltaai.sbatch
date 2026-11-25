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
# This is the batch-queue FALLBACK. The same runs train in 2-hour slices on
# ghx4-interactive (train_llada8b_bl32_dpls_alpha_int_deltaai.sbatch), which starts at once
# while this queue is days out. Both honour $OUTPUT_DIR/RUNNING.lock
# (slurm/lib_cross_cluster_lock.sh): this job exits if TRAIN_DONE exists, re-queues itself
# for an hour if a slice is training, and otherwise takes over from the last checkpoint.
#
# STOP:   scancel -n <job name>

set -euo pipefail

A=${A:?set A=0p5 or A=3}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_bl32_a${A}_dpls
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_alpha${A}_dpls.yaml
SELF=$REPO/slurm/train_llada8b_bl32_dpls_alpha_deltaai.sbatch
CLUSTER=DeltaAI

cd "$REPO"
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }
mkdir -p "$REPO/slurm/logs"
source slurm/lib_cross_cluster_lock.sh
take_lock_or_requeue

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bl32-a${A}-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

# --per_device_train_batch_size 32 keeps 4*32 = generation_batch_size 128.
run_under_lock accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config "$CONFIG" \
  --per_device_train_batch_size 32 \
  --save_steps 10 \
  --output_dir "$OUTPUT_DIR"
