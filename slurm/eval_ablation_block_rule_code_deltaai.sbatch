#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --job-name=ablation-block-rule-code
#SBATCH --output=slurm/logs/%x_%j.out

# Code-side twin of eval_ablation_block_rule_deltaai.sbatch: the KodCode alpha=1 DPLS
# block_unmask policy (llada8b_block_unmask_cond_v2_a1_dpls_kodcode, checkpoint-1064)
# keeps its unmask head and gets its block rule replaced, on HumanEval and MBPP, 0-shot,
# greedy (dpls-greedy unmask + categorical-argmax block), seed 42:
#   J8/J16/J32/J64/J128  constant block size (--block_unmask_fixed_schedule) -> joint_b<b>
#   JADA / JADA16        AdaBlock rule, tau_D 0.3, B0 = 32 / 16                -> joint_ada / joint_ada_b16
#   JT<tau>              learned block head + Fast-dLLM threshold tau (no unmask head) -> joint_thres<tau>
# The learned diagonal is the existing a1 KodCode eval (HumanEval 41.5 @ 71.5,
# MBPP 39.4 @ 49.9 in code/a1_dpls_kodcode and code_mbpp0shot/a1_dpls_kodcode).
# One lane = one cell x one dataset, ~8 (HumanEval) to ~15 (MBPP) min on 2 GH200s:
#   LANE=<cell>:<dataset>, dataset in {humaneval, mbpp}, or a comma-separated list
#   (finished cells leave a marker and are skipped). Pass the list through the
#   environment (LANE=a,b sbatch --export=ALL ...): sbatch splits --export on commas.
# Each job ends by aggregating $RESULTS on the compute node (code grading runs tests).
# Run serially by slurm/logs/ablation_block_rule_code_chain.sh (interactive QoS = 1 job).

set -euo pipefail
REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=${RESULTS:-$WORK/eval_results/ablation_block_rule_code}
CFG=configs/experiment_configs
JOINT_RUN=$WORK/checkpoints/llada8b_block_unmask_cond_v2_a1_dpls_kodcode
JOINT_CFG=$CFG/llada_8b_instruct_dit_block_unmask_joint_cond_v2_alpha1_dpls_kodcode.yaml
# Extra pipeline flags, e.g. EXTRA="--n_test 4" for a smoke.
read -r -a EXTRA <<< "${EXTRA:-}"

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm
export HF_HOME=$WORK/hf_cache
export HF_ALLOW_CODE_EVAL=1
export OMP_NUM_THREADS=8
export EVAL_MAIN_PROCESS_PORT=$((29500 + SLURM_JOB_ID % 400 * 5))
cd "$REPO"
mkdir -p "$RESULTS"

joint() {  # joint <dataset> <subdir> <extra pipeline args...>
  local ds=$1 sub=$2; shift 2
  python -m eval.pipeline "$JOINT_RUN" "$JOINT_CFG" --checkpoints 1064 \
    --remasking block_unmask_policy --sampling_mode dpls-greedy \
    --block_sampling_mode categorical-argmax --datasets "$ds" --few_shot 0 --seeds 42 \
    --temperatures 1.0 --no_aggregate --save_path "$RESULTS/$sub" "$@" "${EXTRA[@]}"
}

run_cell() {  # run_cell <cell> <dataset>
  case "$1" in
    J8|J16|J32|J64|J128) joint "$2" "joint_b${1#J}" --block_unmask_fixed_schedule "${1#J}" ;;
    JADA) joint "$2" joint_ada --adaptive_block --delimiter_threshold 0.3 --block_length 32 ;;
    JADA16) joint "$2" joint_ada_b16 --adaptive_block --delimiter_threshold 0.3 --block_length 16 ;;
    JT0.9|JT0.7|JT0.5) joint "$2" "joint_thres${1#JT}" --block_unmask_threshold "${1#JT}" ;;
    *) echo "unknown cell $1"; exit 2 ;;
  esac
}

echo "=== lane $LANE on $(hostname) $(date)"
for one in ${LANE//,/ }; do
  marker="$RESULTS/.done_${one/:/_}"
  if [ -f "$marker" ]; then echo "=== $one already done, skip"; continue; fi
  run_cell "${one%%:*}" "${one##*:}"
  touch "$marker"
done
python -m eval.aggregate_results --results_dir "$RESULTS" --output_dir "$RESULTS" > /dev/null \
  && echo "=== aggregated" || echo "=== aggregation failed (cells are still on disk)"
echo "=== lane $LANE done $(date)"
