#!/bin/bash
#SBATCH --account=bhta-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=04:00:00
#SBATCH --job-name=ada-b0-sweep-gh
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

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
RESULTS=$WORK/eval_results/adablock_b0_sweep
FIXED_RESULTS=$WORK/eval_results/llada8b_bl32
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml

BASELINE=baseline-fastdllm-t0.9
DELIM_THRES=0.3
B0_LIST="16 32"
# NOTE: the fastdllm baseline is deterministic -- token decoding runs at
# --temperature 0.0 (eval.py's default; eval.pipeline never overrides it) so
# add_gumbel_noise is a no-op, no policy is loaded, and with no --n_test the
# dataset is not subsampled, so nothing consumes the RNG. Seeds 43/44 reproduce
# 42 bit-for-bit. Kept at three seeds because that is what was asked for and it
# costs ~40 extra minutes; set SEEDS=42 to drop them.
SEEDS=42,43,44

module load python/miniforge3_pytorch
eval "$(conda shell.bash hook)"
conda activate dllm

export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=8

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# Smoke phase: both B0 values on 8 samples. Fails fast if the directory naming
# regresses to the name-route collision described above.
SMOKE=$WORK/eval_results/adablock_b0_sweep_smoke
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
