#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=smoke-hidden-proj-gh
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# GH200 counterpart of slurm/smoke_hidden_proj.sbatch: 10 steps per arm for the two
# hidden-state-input policies, measuring peak memory at the batch size DeltaAI would
# actually use.
#
# The point is NOT that GH200 has more memory. DeltaAI has 4 GPUs per node, so
# train_llada8b_bl32_deltaai.sbatch passes --per_device_train_batch_size 32 to keep
# 4*32 = 128 = generation_batch_size, matching the 8-GPU run's 8*16. Per-GPU memory
# depends on pdbs alone, so the batch-matched DeltaAI value is 32 -- twice Delta's 16.
# GH200 has to absorb 2x per card just to run the same experiment.
#
# What the A100-40GB runs measured (jobs 21410797, 21411264), 10 steps each:
#   blocksize_hidden_proj  pdbs 16 -> 28.6 GiB, exit 0
#   hidden_proj            pdbs 16 -> 40.1 GiB, OOM in the rollout
#   hidden_proj            pdbs  8 -> 38.3 GiB, exit 0 with only 2.6 GiB to spare
# Fitting peak ~ 16.0 GiB (weights) + c*pdbs gives c = 0.79 for block_policy and 2.79
# for the per-step path, so at pdbs 32 expect roughly 41 GiB and 105 GiB. That is a
# one-point-per-arm extrapolation and T is data-dependent, so treat it as an order of
# magnitude, not a prediction -- measuring it is the whole point of this job.
#
# Expect blocksize_hidden_proj to pass and hidden_proj to be the interesting one. If it
# OOMs, rerun it alone at the half batch to find where the line is:
#   ARMS=hidden_proj PDBS_PER_STEP=16 sbatch slurm/smoke_hidden_proj_deltaai.sbatch
# pdbs must stay divisible by each config's num_generations (8 per-step, 16 block).
#
# Requires e3f130e or later. HF_HOME is on the shared /work/hdd, so this side resolves
# LLaDA through the same Hub snapshot and hits the same missing final_hidden_state_only
# kwarg that killed job 21410511 on Delta; the signature check that fixes it is in that
# commit. Pull before running.
#
# 1 GPU => accelerate takes the simple launcher: no rendezvous, no port 29500
# collision with co-scheduled jobs (that constraint only bites eval.pipeline).

set -uo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$HF_HOME" "$REPO/slurm/logs"
cd "$REPO"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== commit ==="
git -C "$REPO" log -1 --format='%h %s'

SUMMARY=()

run_arm() {
  local name=$1 config=$2 pdbs=$3
  case " ${ARMS:-hidden_proj blocksize_hidden_proj} " in
    *" $name "*) ;;
    *) return 0 ;;
  esac

  local outdir=$WORK/checkpoints/smoke_gh_$name
  local memlog=$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${name}.mem

  # Fresh dir every time: train.py auto-resumes from get_last_checkpoint(), and a stale
  # smoke checkpoint would silently skip the steps we want to see. It would also fail to
  # load -- these policies have a different state_dict shape. Note the name differs from
  # the Delta script's, because /work/hdd is shared between the two clusters.
  rm -rf "$outdir"
  mkdir -p "$outdir"

  echo
  echo "############ ARM: $name (pdbs=$pdbs) ############"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$memlog" &
  local smi=$!

  accelerate launch --num_processes 1 \
    -m train.train \
    --config "$config" \
    --output_dir "$outdir" \
    --per_device_train_batch_size "$pdbs" \
    --generation_batch_size "$pdbs" \
    --max_steps 10 \
    --save_steps 500 \
    --warmup_steps 2
  local rc=$?

  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  local line="RESULT $name: pdbs=$pdbs exit=$rc peak_gpu_mem=$(sort -n "$memlog" | tail -1) MiB"
  echo "$line"
  SUMMARY+=("$line")
}

run_arm hidden_proj \
  configs/experiment_configs/llada_8b_instruct_dit_hidden_proj_BL32_mixture.yaml \
  "${PDBS_PER_STEP:-32}"

run_arm blocksize_hidden_proj \
  configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint_hidden_proj.yaml \
  "${PDBS_BLOCK:-32}"

echo
echo "=== summary ==="
printf '%s\n' "${SUMMARY[@]}"
