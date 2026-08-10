#!/bin/bash
#SBATCH --account=bhta-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=480g
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-llada8b-bl128-adablock
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sun01024@umn.edu

set -euo pipefail

REPO=/u/zsun9/dllm-sampler-rl
WORK=/work/hdd/bhta/zsun9
CKPT_DIR=$WORK/checkpoints/llada8b_bl128
RESULTS=$WORK/eval_results/llada8b_bl128_adablock
FIXED_RESULTS=$WORK/eval_results/llada8b_bl128
CONFIG=configs/experiment_configs/llada_8b_instruct_dit_confidence_BL128_mixture.yaml

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate dllm

# Keep HF model + dataset caches off the small home quota
export HF_HOME=$WORK/hf_cache
export OMP_NUM_THREADS=16

mkdir -p "$RESULTS" "$REPO/slurm/logs"
cd "$REPO"

# delimiter_threshold 0.3 is the AdaBlock-dLLM (arXiv:2509.26432) LLaDA setting: a
# per-model constant, independent of block size, so it carries over unchanged from
# the BL32 ablation even though block_length is now 128 (a larger max block).

# Smoke phase: both new arms on 8 samples, fails fast before the full runs
SMOKE=$WORK/eval_results/llada8b_bl128_adablock_smoke
rm -rf "$SMOKE"

python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints 1870 \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --sampling_mode bernoulli-argmax \
  --adaptive_block --delimiter_threshold 0.3 \
  --n_test 8 \
  --save_path "$SMOKE" \
  --no_aggregate

python -m eval.pipeline "baseline-fastdllm-t0.9-ada0.3" "$CONFIG" \
  --checkpoints self \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --n_test 8 \
  --save_path "$SMOKE" \
  --no_aggregate

# Smoke check: adaptive runs must record varying block sizes
python - "$SMOKE" <<'EOF'
import glob, json, sys

files = glob.glob(sys.argv[1] + "/**/*_generations.json", recursive=True)
assert len(files) == 2, f"expected 2 smoke result files, got {files}"
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    assert data["adaptive_block"] is True, f
    assert str(data["block_length"]).startswith("ada"), f
    sizes = [g["avg_block_size"] for g in data["generations"]]
    assert all(s is not None for s in sizes), f
    print(f"smoke OK: {f}\n  avg block sizes: {[round(s, 1) for s in sizes]}")
EOF

# Arm D: policy ckpt-1870 + AdaBlock adaptive block sizes, 3 seeds
python -m eval.pipeline "$CKPT_DIR" "$CONFIG" \
  --checkpoints 1870 \
  --datasets gsm8k \
  --seeds 42,43,44 \
  --temperatures 1.0 \
  --sampling_mode bernoulli-argmax \
  --adaptive_block --delimiter_threshold 0.3 \
  --save_path "$RESULTS" \
  --no_aggregate

# Arm C: fastdllm t0.9 + AdaBlock (deterministic decoding, single seed)
python -m eval.pipeline "baseline-fastdllm-t0.9-ada0.3" "$CONFIG" \
  --checkpoints self \
  --datasets gsm8k \
  --seeds 42 \
  --temperatures 1.0 \
  --save_path "$RESULTS" \
  --no_aggregate

# Aggregate the new adaptive arms together with the existing fixed-block results
python -m eval.aggregate_results \
  --results_dir "$FIXED_RESULTS" \
  --results_dir "$RESULTS" \
  --output_dir "$RESULTS"
