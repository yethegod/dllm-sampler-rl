#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=02:00:00
#SBATCH --job-name=diag-blocksize
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# Diagnostics for the joint (block size, threshold) policy, checkpoint-2800.
#
# The GSM8K eval put the learned policy at E[R] = 0.6813 against 0.7238 for the best
# constant (tau=0.7, b=64) and 0.7017 for the AdaBlock heuristic, and the per-position
# action counts looked like a two-phase *schedule*: (64, 0.5) on the first block, then
# tau=0.9 locked in from the second. Realized actions cannot tell a schedule apart from
# genuine per-problem adaptation, because eval samples from the policy. These three
# arms can, and together they cost well under one node-hour:
#
#   1. categorical + recorded action logits -> mutual information between the action
#      and the problem, per decision index. I ~ 0 means the policy is a schedule.
#   2. categorical-argmax -> greedy deployment, as opposed to the E_{a~pi}[R] that
#      GRPO optimises and that arm 1 measures. Separates "expected" from "deployed",
#      and prices the 1.381-nat terminal entropy.
#   3. block_schedule fixed at the modal actions -> the zero-parameter control. If it
#      matches arm 1's E[R], the entire learned policy is reproducible by two constants.
#
# Single seed on purpose: arms 1 and 3 are compared through distributions and E[R]
# whose seed spread is 0.0031, an order of magnitude below the effects in question.
# Results go to a NEW directory so the three-seed run behind the existing report is
# left untouched.
#
# 4 GPUs / whole node: eval.pipeline launches torch.cuda.device_count() processes,
# eval.py's sampler calls dist.get_world_size() unconditionally (so >=2), and the
# rendezvous port 29500 is hardcoded (so two jobs must not share a node).

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_blocksize
RESULTS=$WORK/eval_results/llada8b_blocksize_diag
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml

# Modal per-position actions from the seed-42 eval: (64, 0.5) at decision #0, then
# (32, 0.9) from #1 on (the last entry repeats for every later block).
SCHEDULE="64:0.5,32:0.9"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# 1) Learned policy, sampled -- same setting as the run that produced 0.6813, but the
#    generations now carry per-decision action_logits.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --remasking block_policy \
  --save_path "$RESULTS" \
  --no_aggregate

# 2) Same checkpoint, greedy.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --remasking block_policy \
  --sampling_mode categorical-argmax \
  --save_path "$RESULTS" \
  --no_aggregate

# 3) Fixed-schedule control. The checkpoint path is only used to locate the run
#    directory -- block_schedule ignores the policy weights entirely.
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints last \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --remasking block_schedule \
  --block_schedule "$SCHEDULE" \
  --save_path "$RESULTS" \
  --no_aggregate

# 4) Mutual information per decision index, plus E[R] for all three arms on the same
#    reward the training used. The bar to read E[R] against: 0.7238 best constant,
#    0.7017 AdaBlock, 0.6813 learned policy (3-seed mean), 0.6426 untrained control.
python -m eval.analyze_action_logits "$RESULTS" \
  --alpha 0.5 --gen_length 256 \
  --output "$RESULTS/action_logit_analysis.txt"

echo "Done. Report: $RESULTS/action_logit_analysis.txt"
