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
#SBATCH --job-name=llada8b-bu-dpls-alpha
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# Delta (8xA100) twin of train_llada8b_block_unmask_cond_v2_dpls_alpha_deltaai.sbatch: the
# alpha sweep (A=0p5 or 3) of the v2 joint DPLS block_unmask run, same OUTPUT_DIR and
# wandb run. The config ships per_device_train_batch_size 16 and 8*16 = 128, so no override.
#
# START:  sbatch -J llada8b-bu-a0p5-dpls --export=ALL,A=0p5 slurm/train_llada8b_block_unmask_cond_v2_dpls_alpha.sbatch
#         sbatch -J llada8b-bu-a3-dpls   --export=ALL,A=3   slurm/train_llada8b_block_unmask_cond_v2_dpls_alpha.sbatch
#
# slurm/lib_cross_cluster_lock.sh makes whichever cluster starts first train and the
# other poll hourly; train.train resumes across the 4->8 GPU change on handoff.
#
# STOP:   scancel -n <job name>   (kills running + queued, on THIS cluster only)
# WATCH:  wandb run bu-a${A}-dpls-v1

set -euo pipefail

A=${A:?set A=0p5 or A=3}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a${A}_dpls
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha${A}_dpls.yaml
SELF=$REPO/slurm/train_llada8b_block_unmask_cond_v2_dpls_alpha.sbatch
CLUSTER=Delta

cd "$REPO"
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }
mkdir -p "$REPO/slurm/logs"
source slurm/lib_cross_cluster_lock.sh
take_lock_or_requeue

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a${A}-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

run_under_lock accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml \
  -m train.train \
  --config "$CONFIG" \
  --save_steps 25 \
  --output_dir "$OUTPUT_DIR"
