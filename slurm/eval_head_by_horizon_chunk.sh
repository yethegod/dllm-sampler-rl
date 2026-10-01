#!/bin/bash
# Head x horizon matrix: are horizon-specialised per-position heads (BL32 / BL128,
# alpha=0 reproductions) sensitive to the block size they are run at, unlike the
# horizon-robust v1 block_unmask head (trained under random block sizes)?
# Rows = head, columns = block_length at eval. The diagonal cells (BL32@32,
# BL128@128, 3 seeds) already exist in eval_results/llada8b_bl{32,128}; this adds
# the off-diagonal ones into eval_results/head_by_horizon/<head>/.
# Usage: HEAD=bl32|bl128 BLOCKS="8 128" SEEDS=42 bash slurm/eval_head_by_horizon_chunk.sh
set -euo pipefail
HEAD=${HEAD:?bl32|bl128}
BLOCKS=${BLOCKS:-"8 32 128"}
SEEDS=${SEEDS:-42}
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_$HEAD
RESULTS=$WORK/eval_results/head_by_horizon/$HEAD
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_${HEAD^^}_mixture.yaml
module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
mkdir -p "$RESULTS"
cd "$REPO"
for b in $BLOCKS; do
  echo "=== head $HEAD at block_length=$b seeds $SEEDS $(date)"
  python -m eval.pipeline "$CKPT_DIR" "$CONFIG" --checkpoints last --datasets gsm8k \
    --seeds "$SEEDS" --temperatures 1.0 --remasking policy --sampling_mode bernoulli-argmax \
    --block_length "$b" --record_unmask_order --save_path "$RESULTS" --no_aggregate
done
echo "=== done $(date)"
