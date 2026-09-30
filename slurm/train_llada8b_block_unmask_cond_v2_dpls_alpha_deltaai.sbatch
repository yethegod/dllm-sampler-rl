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
#SBATCH --job-name=llada8b-bu-dpls-alpha
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# The alpha sweep of the v2 joint DPLS block_unmask run, DeltaAI side (4xGH200): the
# alpha=1 run (train_llada8b_block_unmask_cond_v2a1_dpls_deltaai.sbatch) at alpha 0.5 or 3.
# A picks the config, OUTPUT_DIR and wandb run:
#   configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha${A}_dpls.yaml
#
# START:  sbatch -J llada8b-bu-a0p5-dpls --export=ALL,A=0p5 slurm/train_llada8b_block_unmask_cond_v2_dpls_alpha_deltaai.sbatch
#         sbatch -J llada8b-bu-a3-dpls   --export=ALL,A=3   slurm/train_llada8b_block_unmask_cond_v2_dpls_alpha_deltaai.sbatch
#
# The same run may also be queued on Delta (train_llada8b_block_unmask_cond_v2_dpls_alpha.sbatch);
# slurm/lib_cross_cluster_lock.sh makes whichever starts first train and the other poll
# hourly. The alpha=1 run fit its 1870-step epoch in one 48h job here.
#
# STOP:   scancel -n <job name>   (kills running + queued, on THIS cluster only)
# WATCH:  wandb run bu-a${A}-dpls-v1: block_size/frac_*, block_entropy, tokens_per_step_mean,
#         num_steps_mean, still_masked == 0.

set -euo pipefail

A=${A:?set A=0p5 or A=3}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a${A}_dpls
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha${A}_dpls.yaml
SELF=$REPO/slurm/train_llada8b_block_unmask_cond_v2_dpls_alpha_deltaai.sbatch
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
# One wandb run per alpha, shared by both clusters and every restart.
export WANDB_RUN_ID=${WANDB_RUN_ID:-bu-a${A}-dpls-v1}
export WANDB_RESUME=allow
export OMP_NUM_THREADS=16

# --per_device_train_batch_size 32 keeps 4*32 = generation_batch_size 128 (see the
# alpha=1 sbatch). --save_steps 25 halves what a cross-cluster handoff can lose.
run_under_lock accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config "$CONFIG" \
  --per_device_train_batch_size 32 \
  --save_steps 25 \
  --output_dir "$OUTPUT_DIR"
