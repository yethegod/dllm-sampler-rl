#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=ablation-block-rule
#SBATCH --output=slurm/logs/%x_%j.out

# Ablation grid (unmask head x block rule) on GSM8K and MATH500, greedy, seed 42:
#
#                         fixed b=32     AdaBlock B0=32    learned block head (joint a1)
#   joint a1 unmask head  J32            JADA              existing (78.70 / 35.4)
#   fixed b=32 head       F32            FADA              FLRN
#
# joint a1 = llada8b_block_unmask_cond_v2_a1_dpls ckpt-1870; fixed head =
# llada8b_bl32_a1_dpls (dit_confidence, DPLS, alpha=1), last checkpoint unless FIX_CKPT.
# FLRN runs the joint policy for block decisions and the fixed head for unmasking
# (--unmask_policy_path). One lane = one cell x one dataset, ~20-50 min on 2 GH200s:
#   LANE=<cell>:<dataset>, e.g. LANE=JADA:gsm8k, dataset in {gsm8k, math}
#   LANE=SMOKE  n_test 4 of every new path, plus an exactness check: FLRN with the
#               block forced to 32 must equal F32.
# Run serially by slurm/logs/ablation_block_rule_chain.sh (interactive QoS = 1 job).

set -euo pipefail
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/ablation_block_rule}
CFG=configs/experiment_configs
JOINT_RUN=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls
JOINT_CFG=$CFG/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls.yaml
FIX_RUN=$WORK/checkpoints/llada8b_bl32_a1_dpls
FIX_CFG=$CFG/llada_8b_instruct_dit_confidence_BL32_alpha1_dpls.yaml

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + SLURM_JOB_ID % 400 * 5))
cd "$REPO"

FIX_CKPT=${FIX_CKPT:-$(ls -d "$FIX_RUN"/checkpoint-[0-9]* | sed 's/.*checkpoint-//' | sort -n | tail -1)}
EXTRA=()
if [ "$LANE" = SMOKE ]; then
  RESULTS=$WORK/eval_results/ablation_block_rule_smoke
  EXTRA=(--n_test 4)
fi
UPOL=$FIX_RUN/checkpoint-$FIX_CKPT/model.safetensors
[ -f "$UPOL" ] || { echo "missing $UPOL"; exit 2; }
mkdir -p "$RESULTS"

joint() {  # joint <dataset> <subdir> <extra pipeline args...>
  local ds=$1 sub=$2; shift 2
  python -m eval.pipeline "$JOINT_RUN" "$JOINT_CFG" --checkpoints 1870 \
    --remasking block_unmask_policy --sampling_mode dpls-greedy \
    --block_sampling_mode categorical-argmax --datasets "$ds" --seeds 42 \
    --temperatures 1.0 --no_aggregate --save_path "$RESULTS/$sub" "$@" "${EXTRA[@]}"
}
fixed() {  # fixed <dataset> <subdir> <extra pipeline args...>
  local ds=$1 sub=$2; shift 2
  python -m eval.pipeline "$FIX_RUN" "$FIX_CFG" --checkpoints "$FIX_CKPT" \
    --remasking policy --sampling_mode dpls-greedy --block_length 32 --datasets "$ds" \
    --seeds 42 --temperatures 1.0 --no_aggregate --save_path "$RESULTS/$sub" "$@" "${EXTRA[@]}"
}
ADA=(--adaptive_block --delimiter_threshold 0.3 --block_length 32)
UPOL_ARGS=(--unmask_policy_path "$UPOL" --unmask_policy_config "$FIX_CFG")

run_cell() {  # run_cell <cell> <dataset>
  case "$1" in
    J32)  joint "$2" joint_b32 --block_unmask_fixed_schedule 32 ;;
    JADA) joint "$2" joint_ada "${ADA[@]}" ;;
    F32)  fixed "$2" fixed_b32 ;;
    FADA) fixed "$2" fixed_ada "${ADA[@]}" ;;
    FLRN) joint "$2" fixed_learned "${UPOL_ARGS[@]}" ;;
    *) echo "unknown cell $1"; exit 2 ;;
  esac
}

echo "=== lane $LANE on $(hostname) $(date), fixed head ckpt $FIX_CKPT"
if [ "$LANE" = SMOKE ]; then
  rm -rf "$RESULTS"
  for c in JADA FADA FLRN F32; do run_cell "$c" gsm8k; done
  joint gsm8k fixed_learned_fixed32 "${UPOL_ARGS[@]}" --block_unmask_fixed_schedule 32
  python - "$RESULTS" <<'EOF'
import glob, json, sys
root = sys.argv[1]
def load(sub):
    (f,) = glob.glob(f"{root}/{sub}/**/*_generations.json", recursive=True)
    return json.load(open(f))
for sub in ("joint_ada", "fixed_ada", "fixed_learned", "fixed_b32", "fixed_learned_fixed32"):
    d = load(sub)
    g = d["generations"]
    print(sub, d["block_length"], "n", len(g), "NFE", [x["steps"] for x in g],
          "blocks", [x["block_sizes"] for x in g][:2])
a, b = load("fixed_b32")["generations"], load("fixed_learned_fixed32")["generations"]
key = lambda x: x["question"]
a, b = sorted(a, key=key), sorted(b, key=key)
same = [x["generations"] == y["generations"] and x["steps"] == y["steps"] for x, y in zip(a, b)]
print("exactness F32 vs FLRN-forced-32:", same)
sys.exit(0 if len(a) == len(b) == 4 and all(same) else 1)
EOF
else
  run_cell "${LANE%%:*}" "${LANE##*:}"
fi
echo "=== lane $LANE done $(date)"
