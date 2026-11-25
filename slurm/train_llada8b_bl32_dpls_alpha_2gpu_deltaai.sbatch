#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=llada8b-bl32-dpls-alpha
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,REQUEUE
#SBATCH --mail-user=sun01024@umn.edu

# 2-GPU shape of train_llada8b_bl32_dpls_alpha_deltaai.sbatch (see its header): same run,
# per-device batch 64, about half the step rate (~46h for the epoch, so it may need one
# re-queue). It fits a partly used node, which a full ghx4 hands out far sooner than a
# whole one. A 4-GPU job that starts later cancels this one and resumes its checkpoints.
# --mem=120g: nodes have 474 GB for 4 GPUs and most jobs take the 113 GB/GPU default, so a
# node with two free GPUs often has well under 240 GB left; the 4-GPU run peaked at 81 GB.
#
# START:  sbatch -J llada8b-bl32-a3-dpls --export=ALL,A=3 slurm/train_llada8b_bl32_dpls_alpha_2gpu_deltaai.sbatch

REPO=/u/zsun9/dllm-sampler-rl
export SELF=$REPO/slurm/train_llada8b_bl32_dpls_alpha_2gpu_deltaai.sbatch
# exec keeps the PID, so the USR1 / TERM that Slurm sends the batch shell reach the traps.
exec bash "$REPO/slurm/train_llada8b_bl32_dpls_alpha_deltaai.sbatch"
