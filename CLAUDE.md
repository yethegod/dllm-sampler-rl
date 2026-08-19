# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of Apple's release for *Learning Unmasking Policies for Diffusion Language Models* (arXiv:2512.09106). A frozen diffusion LLM (LLaDA-8B-Instruct or Dream-7B) is the environment; a small DiT-style policy learns, via GRPO, which masked positions to unmask at each denoising step. Only the policy is trained — the dLLM weights are never updated.

Local additions on top of upstream (see `git log`): AdaBlock-style adaptive block sizes at eval time, SLURM scripts for NCSA Delta / DeltaAI, resume + multi-GPU-reshape fixes, and `eval/plot_pareto_frontier.py`. Upstream files carry Apple license headers — preserve them when editing.

## Environment & commands

Python 3.12, conda env `dllm`. There is no test suite, linter config, or CI in this repo — do not invent commands for them. `HF_TOKEN` is read from `.env` in the working directory (gitignored).

**Commit to `main`.** This is a single-author fork with no PR flow; do not open a feature branch for a change unless explicitly asked. The two clusters do not share home directories, so `main` on `origin` is how code moves between Delta and DeltaAI — work parked on a side branch is work the other cluster cannot run.

```bash
pip install -e .

# Train (single node, multi-GPU via accelerate DDP)
accelerate launch --config_file configs/accelerate_configs/8gpu_ddp.yaml -m train.train \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --output_dir /work/hdd/bhta/zsun9/checkpoints/llada8b_bl32

# Evaluate a run directory of checkpoint-* dirs (drives eval.eval as subprocesses)
python -m eval.pipeline <ckpt_dir> <config.yaml> --checkpoints last --datasets gsm8k \
  --seeds 42,43,44 --temperatures 1.0 --sampling_mode bernoulli-argmax --save_path <results>

# Aggregate JSON generations -> detailed_results.csv / summary_statistics.csv / results_report.txt
python -m eval.aggregate_results --results_dir <results> [--results_dir <more>] [--output_dir <dir>]

python eval/plot_pareto_frontier.py   # reads summary_statistics.csv, writes eval/plots/
```

Real runs go through `slurm/*.sbatch` (`sbatch slurm/train_llada8b_bl32.sbatch`). Those scripts are the source of truth for cluster conventions; copy an existing one rather than writing a new preamble.

## Architecture

**Single decoding loop, two callers.** `common/generation/generation.py:generate_unified` implements every decoding strategy and is used by *both* training rollouts (`train/trainer.py:_generate_and_score_completions`) and eval (`eval/eval.py:evaluate`). Any change to decoding affects both paths. `remasking` selects the strategy: `policy` (learned per-position), `block_policy` (learned joint block size + threshold, `sampling_mode` `categorical` for training or `categorical-argmax` for greedy eval), `block_schedule` (eval-only control: the same loop driven by a fixed `--block_schedule "b:thres,..."` list instead of a policy, last entry repeating), `fastdllm` (confidence threshold `thres`), `low_confidence` / `random` (fixed steps-per-block baselines). When `record_policy_data` is on it also returns the per-timestep policy inputs, samples, and masks needed for the loss.

**Policy** (`common/models/policy.py`): `DiTConfidencePolicy` (main; sees top-p token confidences, ~100k params) or `DiTHiddenStatePolicy` (ablation; LLaDA only). Wrapped in `PolicyHFWrapper` so HF Trainer machinery (checkpointing, DDP) treats it as the model. Sampling of the policy logits lives in `common/generation/sampling.py`: `bernoulli` / `dpls` for training, `bernoulli-argmax` for eval of a bernoulli-trained policy.

**Trainer** (`train/trainer.py`, subclass of TRL `GRPOTrainer`): the GRPO "token" is an unmasking *timestep*, not a text token. `compute_loss` walks a list of per-batch policy outputs (T varies across group members, so they cannot be stacked), recomputes per-timestep logprobs in chunks of `timestep_batch_size`, and applies the clipped ratio against group advantages. `beta` must be 0 (no KL term implemented). Optional expert-steering (`es_thresholds`) mixes dirac rollouts into the group.

**Reward** (`train/reward_func.py`): `mixed_correctness_mult_reward_func` dispatches per-sample on `dataset_type` (gsm8k/math/kodcode) and multiplies correctness by a compute term controlled by `alpha_compute_reward` — higher alpha buys speed at the cost of accuracy. Reward functions are resolved by name from the config's `reward_functions` list.

**Config**: `common/config.py:Config` extends TRL `GRPOConfig` with all diffusion/policy fields; everything comes from the YAML in `configs/experiment_configs/`, overridable by CLI flags via `TrlParser`. `configs/accelerate_configs/` holds the DDP launch configs (4- and 8-GPU).

**Eval flow**: `eval.pipeline` resolves checkpoints (`first`/`last`/numbers/`best`), fans out over checkpoints × datasets × seeds × temperatures, and shells out to `accelerate launch -m eval.eval` per combination with `--batch_size 1`; each writes `<dataset>_generations.json` into a directory whose *name* encodes the run parameters. `eval.aggregate_results` re-parses those directory names, so the naming scheme in `eval/pipeline.py:run_eval` and the parsers in `aggregate_results.py` must stay in sync.

## Invariants that bite

- `generation_batch_size` must equal `per_device_train_batch_size * num_processes`, otherwise TRL silently sets `steps_per_generation > 1` and the input buffering / advantage slicing desynchronize. `train/trainer.py` asserts this. All configs ship `generation_batch_size: 128` with `per_device_train_batch_size: 16` (i.e. 8 GPUs); the 4-GPU sbatch scripts pass `--per_device_train_batch_size 32` to keep the product at 128.
- `gradient_accumulation_steps` must be 1; `per_device_train_batch_size` must be divisible by `num_generations`.
- Training requires `remasking: policy`; mbpp/humaneval are eval-only datasets.
- Resume is automatic: `train.train` calls `get_last_checkpoint(output_dir)` (or pulls the latest from S3 when `output_dir` starts with `s3://`). `Trainer._inner_training_loop` deliberately ignores the batch size restored from `trainer_state.json` so a run can resume across a GPU-count change.
- **Baselines are encoded in checkpoint names**, parsed by `eval/eval.py:parse_baseline_checkpoint`: `baseline-<method>[-K<steps>][-t<thres>][-ada<threshold>]`, passed as the run path with `--checkpoints self` (e.g. `"baseline-low_confidence-K256;baseline-fastdllm-t0.9"`, semicolon-separated). Adding a knob means touching that regex, the pipeline flag, and the output-dir naming together.
- Adaptive block (`--adaptive_block`) requires batch size 1, `full_context=True` for policy remasking, and `remasking` in {`policy`, `fastdllm`}. Its boundary decision reuses the block's first forward pass, so it costs no extra NFE. Results record `block_length` as the string `ada<threshold>_B<B0>` (and the output dir carries the same suffix) so adaptive and fixed-block runs don't collide during aggregation, and neither do two adaptive runs at different B0. `delimiter_threshold` (0.3 for LLaDA) is a per-model constant, independent of block size, but B0 is *not* inert: it is the fallback when no delimiter clears the threshold, and the lookahead window is a fixed `0.25 * gen_length`, so the ratio `0.25*L / B0` decides whether a block can ever grow past B0.

## Cluster notes (NCSA)

Two clusters share `/work/hdd/bhta/zsun9` (checkpoints, `HF_HOME`, eval results) but **not** home directories — the repo and conda env exist separately on each side.

- **Delta**: `--account=bhta-delta-gpu --partition=gpuA100x8`, 8×A100, x86. Conda via `source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh`.
- **DeltaAI**: `--account=bhta-dtai-gh --partition=ghx4`, 4×GH200, **aarch64** — `pyproject.toml` pins `bitsandbytes>=0.46.0` there because earlier versions have no aarch64 wheels. Conda via `module load python/miniforge3_pytorch`.

Because `/work/hdd` is shared, a BL32 run started on Delta can be resumed on DeltaAI against the same `OUTPUT_DIR`.
