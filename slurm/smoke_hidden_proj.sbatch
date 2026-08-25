#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --job-name=smoke-hidden-proj
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# 10-step single-GPU smoke test for the two hidden-state-input policies (10f36eb).
# Everything so far was verified against a toy LLaDA on the login node; this is the
# first run of these policies against the real 8B model. What it must show:
#   - both decode paths run and the loss backward reaches the projector
#   - peak GPU memory near the estimate: ~25.5 GiB for the per-step path (16.0
#     weights + 2.9 logits + 1.9 all-33-layer hidden states + 4.2 policy_inputs
#     buffer + 0.5 chunked fp32 cast), and less for block_policy, whose rec_c is
#     capped at T_cap=32 decisions rather than T~134 forwards
#   - timestep_batch_size:8 is actually taking effect: unset, compute_loss casts the
#     whole (B,T,L,4096) buffer to fp32 at once, +8 GiB, which would not fit here.
#     The 1.9 GiB for the unused per-layer hidden states is NOT saved -- LLaDA loads
#     from the Hub's modeling_llada, which has no final_hidden_state_only (job
#     21410511 crashed on exactly that); generate_unified now signature-checks it.
#
# Per-GPU memory depends on per_device_train_batch_size, not on world size, so one GPU
# measures what each of the 8 would use. generation_batch_size tracks pdbs to keep
# gbs == pdbs * num_processes on one GPU (the trainer asserts this). Each config keeps
# its own num_generations (8 per-step, 16 block_policy), and both divide their pdbs.
# ARMS='<names>' sbatch ... selects arms; PDBS_PER_STEP / PDBS_BLOCK override the
# per-arm batch size. Default runs all four; the two chunk_control_* arms are the A/B
# for the chunk-wise backward and are cheap.
#
# 1 GPU => accelerate takes the simple launcher: no rendezvous, no port 29500
# collision with co-scheduled jobs (that constraint only bites eval.pipeline).

set -uo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
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
  local name=$1 config=$2 pdbs=$3 extra=${4:-}
  # Keep in sync with the run_arm calls below -- an arm missing from this default
  # is silently skipped, which is how job 21447593 ran two arms instead of four.
  case " ${ARMS:-hidden_proj blocksize_hidden_proj chunk_control_off chunk_control_on} " in
    *" $name "*) ;;
    *) return 0 ;;
  esac
  local outdir=$WORK/checkpoints/smoke_$name
  local memlog=$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${name}.mem

  # Fresh dir every time: train.py auto-resumes from get_last_checkpoint(), and a
  # stale smoke checkpoint would silently skip the steps we want to see. It would
  # also fail to load -- these policies have a different state_dict shape.
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
    --warmup_steps 2 \
    $extra
  local rc=$?

  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  local line="RESULT $name: pdbs=$pdbs exit=$rc peak_gpu_mem=$(sort -n "$memlog" | tail -1) MiB"
  echo "$line"
  SUMMARY+=("$line")
}

# Back to the real pdbs 16, which OOM'd at 40.1 GiB before replay (job 21410797): the
# rollout no longer buffers (B,T,L,4096) hidden states, and compute_loss backwards each
# timestep chunk instead of holding all of them for one backward at the end. Expect the
# peak to land near 25 GiB rather than the ~56 GiB the old path needed here.
run_arm hidden_proj \
  configs/experiment_configs/llada_8b_instruct_dit_hidden_proj_BL32_mixture.yaml \
  "${PDBS_PER_STEP:-16}"

run_arm blocksize_hidden_proj \
  configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint_hidden_proj.yaml \
  "${PDBS_BLOCK:-16}"

# Control pair. Chunk-wise backward changes the order gradients accumulate in, so it is
# mathematically equivalent but not bit-identical, and it now applies to the confidence
# policies every existing run uses. These two differ ONLY in the number of chunks
# (unset => one chunk => the pre-change path; 8 => ~32 chunks => the new one), same
# config and same seed, so any drift in loss/grad_norm between them is the change
# itself rather than run-to-run noise.
run_arm chunk_control_off \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml 16

run_arm chunk_control_on \
  configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml 16 \
  "--timestep_batch_size 8"

echo
echo "=== summary ==="
printf '%s\n' "${SUMMARY[@]}"
