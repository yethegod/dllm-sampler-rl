#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=04:00:00
#SBATCH --job-name=llada8b-block-unmask-cond-v2a0-chunk
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sun01024@umn.edu

# The alpha=0 v2 joint run (see train_llada8b_block_unmask_cond_v2a0_deltaai.sbatch
# and the config header) cut into 4h backfill-sized chunks. The 20h version sat
# in the ghx4 queue for >1 day (job 3173188, start estimate kept slipping); 4h is
# the most common pending time limit on ghx4 and is what backfill actually runs.
#
# HOW THE CHAIN WORKS
#   1. At start, this job submits one more copy of itself with
#      --dependency=afterany:<this job>, so the next chunk is queued (and starts
#      accruing priority) while this one runs. afterany, not afterok: a chunk that
#      hits its time limit exits non-zero and must still be followed.
#   2. train.train auto-resumes from the latest checkpoint-* in OUTPUT_DIR; with
#      --save_steps 25 (15 MB each) a chunk loses at most ~25 steps on timeout.
#   3. When accelerate exits 0 the epoch is complete: TRAIN_DONE is written and
#      every later chunk exits at once, so the chain ends by itself.
#   4. CHUNK_INDEX (exported to each child) caps the chain at MAX_CHUNKS so a
#      persistent crash cannot resubmit forever. Raise MAX_CHUNKS if the run is
#      slower than expected and legitimately needs more chunks.
#
# START:  sbatch slurm/train_llada8b_block_unmask_cond_v2a0_chunk_deltaai.sbatch
# STOP:   scancel -n llada8b-block-unmask-cond-v2a0-chunk   (kills running + queued)
# WATCH:  python -m eval.monitor_block_head $WORK/checkpoints/llada8b_block_unmask_cond_v2_a0
#
# Same watch-list as the a1 run: block_size/frac_* and block_entropy for the
# block head, mean_unmask_prob -> 0 or NaN loss as the kill signals. A slow
# policy (num_steps_mean > 100) is the expected shape at alpha=0, not a failure.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
OUTPUT_DIR=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a0
DONE_MARKER=$OUTPUT_DIR/TRAIN_DONE

CHUNK_INDEX=${CHUNK_INDEX:-1}
MAX_CHUNKS=${MAX_CHUNKS:-14}   # 14 x 4h = 56h of wall clock, ~2x the a1 pace

cd "$REPO"
mkdir -p "$OUTPUT_DIR" "$REPO/slurm/logs"

if [ -f "$DONE_MARKER" ]; then
  echo "[chunk $CHUNK_INDEX] $DONE_MARKER exists; training already complete. Exiting."
  exit 0
fi

echo "[chunk $CHUNK_INDEX/$MAX_CHUNKS] job $SLURM_JOB_ID on $(hostname), $(date)"
echo "latest checkpoint: $(ls -d "$OUTPUT_DIR"/checkpoint-[0-9]* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || echo none)"

# Queue the next chunk before doing anything slow.
if [ "$CHUNK_INDEX" -lt "$MAX_CHUNKS" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,CHUNK_INDEX=$((CHUNK_INDEX + 1)),MAX_CHUNKS="$MAX_CHUNKS" \
    "$REPO/slurm/train_llada8b_block_unmask_cond_v2a0_chunk_deltaai.sbatch")
  echo "queued chunk $((CHUNK_INDEX + 1)) as job $NEXT (afterany:$SLURM_JOB_ID)"
else
  echo "MAX_CHUNKS=$MAX_CHUNKS reached; not queueing a successor."
fi

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16
mkdir -p "$HF_HOME"

# --per_device_train_batch_size 32 is mandatory on 4 GPUs (4*32 = generation_batch_size
# 128, keeps steps_per_generation at 1; train/trainer.py asserts it).
# --save_steps 25 overrides the config's 50 so a timed-out chunk loses less.
accelerate launch --config_file configs/accelerate_configs/4gpu_ddp.yaml \
  -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha0.yaml \
  --per_device_train_batch_size 32 \
  --save_steps 25 \
  --output_dir "$OUTPUT_DIR"

# Only reached if accelerate exited 0, i.e. trainer.train() returned: epoch done.
date > "$DONE_MARKER"
echo "[chunk $CHUNK_INDEX] training complete; wrote $DONE_MARKER"
