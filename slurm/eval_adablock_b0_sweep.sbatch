#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --job-name=ada-b0-sweep
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

# AdaBlock-dLLM (arXiv:2509.26432) B0 sweep at the paper's Fast-dLLM threshold.
#
# The paper reports B0 in {16,32,64}; we had only two Ada cells, and they moved
# B0 and thres together (B0=32/t0.9 and B0=64/t0.7), so neither isolates B0.
# This fills the t0.9 row at B0 in {16,32}. B0=64/t0.9 is the remaining hole.
#
# WHY THE FLAG ROUTE, NOT THE NAME ROUTE. There are two ways to make a baseline
# adaptive, and only one of them names the output directory correctly:
#   name route: "baseline-fastdllm-t0.9-ada0.3" -> eval.py's
#     parse_baseline_checkpoint sets adaptive_block, but eval.pipeline's own
#     cfg.adaptive_block stays False, so the "_ada<tau_D>_B<B0>" suffix added in
#     137d0a6 is never appended and *two B0 values overwrite each other*.
#   flag route (used here): "baseline-fastdllm-t0.9" plus --adaptive_block
#     --delimiter_threshold --block_length. cfg.adaptive_block is True, so
#     resolve_block_length() puts B0 in the directory name and eval.py records
#     block_length as "ada0.3_B<B0>". Baseline-name parsing never touches
#     block_length, so the --block_length override survives.
# Do not "simplify" this back to the name route.
#
# The existing B0=32 cell in eval_results/llada8b_bl32_adablock predates 137d0a6
# and is labelled plain "ada0.3" with no B0, so it cannot be merged with these
# rows or even identified by B0 after the fact. It is re-run here rather than
# reused; a cell is ~20 min on 4 GPUs, so reuse is not worth the ambiguity.

# Delta (gpuA100x4), not DeltaAI: the ghx4 copy of this sweep sat behind a
# ~29h priority wait. Same shared /work/hdd paths, so the results land in the
# same place either way; only the preamble and the conda hook differ.
# 4h walltime: 2 cells at ~50 min each on 4xA100 (the 8h figure was for the
# 3-seed, 6-cell version). Sized for backfill, not for headroom -- if a cell
# turns out slower than that, the finished cells are already on disk and the
# rest can be resubmitted.

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=$WORK/eval_results/adablock_b0_sweep
FIXED_RESULTS=$WORK/eval_results/llada8b_bl32
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml

BASELINE=baseline-fastdllm-t0.9
DELIM_THRES=0.3
B0_LIST="16 32"
# NOTE: one seed only, deliberately. The fastdllm baseline is deterministic --
# token decoding runs at --temperature 0.0 (eval.py's default; eval.pipeline
# never overrides it) so add_gumbel_noise is a no-op, no policy is loaded, and
# with no --n_test the dataset is not subsampled, so nothing consumes the RNG.
# Seeds 43/44 reproduced 42 bit-for-bit, i.e. ~3.3h of duplicate decoding on
# 4xA100 for zero extra information, so they were dropped (job 21325233 was
# cancelled mid-flight for this and resubmitted). Do not add seeds back here
# unless something in this cell stops being deterministic -- a learned policy
# (remasking: policy with a sampling temperature), --n_test subsampling, or a
# nonzero --temperature. Error bars for these rows have to come from bootstrap
# over the 1319 examples, not from seeds.
SEEDS=42

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# Smoke phase: both B0 values on 8 samples. Fails fast if the directory naming
# regresses to the name-route collision described above. It costs ~40 min,
# almost all of it re-reading the 15GB of LLaDA weights off /work/hdd twice, so
# SKIP_SMOKE=1 is there for a resubmit of unchanged code whose smoke phase
# already passed in an earlier job. Leave it at 0 whenever this script, the
# pipeline naming, or eval.py's block_length labelling has been touched.
SKIP_SMOKE=${SKIP_SMOKE:-0}
SMOKE=$WORK/eval_results/adablock_b0_sweep_smoke

if [ "$SKIP_SMOKE" = "1" ]; then
  echo "SKIP_SMOKE=1: skipping the 8-sample naming check"
else
rm -rf "$SMOKE"

for B0 in $B0_LIST; do
  python -m eval.pipeline "$BASELINE" "$CONFIG" \
    --checkpoints self \
    --datasets gsm8k \
    --seeds 42 \
    --temperatures 1.0 \
    --block_length "$B0" \
    --adaptive_block --delimiter_threshold "$DELIM_THRES" \
    --n_test 8 \
    --save_path "$SMOKE" \
    --no_aggregate
done

python - "$SMOKE" "$DELIM_THRES" <<'EOF'
import glob, json, sys

smoke, delim = sys.argv[1], sys.argv[2]
files = glob.glob(f"{smoke}/**/*_generations.json", recursive=True)
assert files, f"no generations under {smoke}"
labels = set()
for f in files:
    data = json.load(open(f))
    assert data["adaptive_block"] is True, f
    label = str(data["block_length"])
    assert label.startswith(f"ada{delim}_B"), f"{f}: block_length={label!r}, B0 missing"
    labels.add(label)
    sizes = {b for r in data["generations"] for b in r["block_sizes"]}
    assert len(sizes) > 1, f"{f}: block sizes never varied ({sizes})"
# One directory per B0, not one shared directory.
assert len(labels) == len(files), f"B0 cells collided: {labels} across {len(files)} files"
print(f"smoke OK: {sorted(labels)}")
EOF
fi

# Full sweep.
for B0 in $B0_LIST; do
  python -m eval.pipeline "$BASELINE" "$CONFIG" \
    --checkpoints self \
    --datasets gsm8k \
    --seeds "$SEEDS" \
    --temperatures 1.0 \
    --block_length "$B0" \
    --adaptive_block --delimiter_threshold "$DELIM_THRES" \
    --save_path "$RESULTS" \
    --no_aggregate
done

# Aggregate alongside the fixed-block BL32 grid so the Ada rows sit next to the
# constant-b bars they have to beat.
python -m eval.aggregate_results \
  --results_dir "$RESULTS" \
  --results_dir "$FIXED_RESULTS" \
  --output_dir "$RESULTS"
