#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240g
#SBATCH --time=01:00:00
#SBATCH --job-name=smoke-a1-dpls-kodcode-2gpu
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sun01024@umn.edu

# 3-step, 2-GPU DDP smoke of the KodCode a1 DPLS run at its real per-device batch
# (32, as on 4xGH200; gbs 64 = 32 x 2 here). Checks: KodCode load + drop_long_prompts
# count, gen_prefix prompts up to 512 tokens, code_eval reward on compute nodes,
# DDP gathers, and peak memory at B=32 with the longer prompt.
# Fresh output dir; never the real OUTPUT_DIR.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/smoke_block_unmask_a1_dpls_kodcode_2gpu

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

rm -rf "$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$OUTPUT_DIR" "$REPO/slurm/logs"
cd "$REPO"

# Peak-memory sampler: one line per GPU every 15 s; the max is printed at the end.
MEMLOG=$OUTPUT_DIR/gpu_mem.csv
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 15 > "$MEMLOG" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

echo "[smoke] start $(date) on $(hostname)"
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  --num_processes 2 --main_process_port 29541 \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 32 \
  --generation_batch_size 64 \
  --max_steps 3 \
  --save_steps 500 \
  --warmup_steps 2
echo "[smoke] end $(date)"

kill $SMI_PID 2>/dev/null || true
echo "[smoke] peak GPU memory used (MiB) per GPU:"
awk -F', ' '{ if ($2+0 > mx[$1]) mx[$1]=$2+0; tot[$1]=$3 } END { for (g in mx) printf "  GPU %s: %d / %d MiB\n", g, mx[g], tot[g] }' "$MEMLOG"
