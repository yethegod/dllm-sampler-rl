#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Evaluation pipeline: download checkpoint, evaluate, aggregate."""

import argparse
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import s3fs
import torch
import yaml


@dataclass
class EvalConfig:
    run_path: str
    config_path: str
    datasets: list[str]
    temperatures: list[float]
    sampling_mode: str | None
    checkpoints: list[str]
    seeds: list[int]
    block_length: int | None
    gen_length: int | None
    model_path: str
    save_path: str
    n_test: int | None
    adaptive_block: bool = False
    delimiter_threshold: float = 0.3
    remasking: str = "policy"
    block_schedule: str | None = None


def resolve_block_length(cfg: EvalConfig) -> int:
    """B0 for an adaptive-block run: the CLI override, else the config's value.

    eval.eval resolves it the same way; we duplicate it here because the output
    directory name is built before the subprocess runs.
    """
    if cfg.block_length is not None:
        return cfg.block_length
    with open(cfg.config_path) as f:
        block_length = yaml.safe_load(f).get("block_length")
    assert block_length is not None, (
        f"{cfg.config_path} sets no block_length; --adaptive_block needs it to "
        "name the output dir, so pass --block_length explicitly"
    )
    return int(block_length)


def get_s3() -> s3fs.S3FileSystem:
    raise NotImplementedError("Internal bucket setup stripped.")


def download_checkpoint(
    s3: s3fs.S3FileSystem, s3_path: str, checkpoint: str, local_dir: Path
):
    if checkpoint.startswith("baseline-"):
        ckpt_dir = local_dir / f"checkpoint-{checkpoint}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / ".baseline_marker").touch()
        return

    # Strip s3:// prefix for s3fs
    s3_path_clean = s3_path.replace("s3://", "")
    s3_ckpt = f"{s3_path_clean}/checkpoint-{checkpoint}"
    local_ckpt = local_dir / f"checkpoint-{checkpoint}"

    if local_ckpt.exists() and (local_ckpt / "model.safetensors").exists():
        print(f"  Checkpoint {checkpoint} already downloaded")
        return

    print(f"  Downloading checkpoint-{checkpoint}...")
    local_dir.mkdir(parents=True, exist_ok=True)
    # s3fs.get() with recursive=True creates the target directory
    s3.get(s3_ckpt, str(local_ckpt), recursive=True)

    # s3fs may create nested directory - check and fix
    nested = local_ckpt / f"checkpoint-{checkpoint}"
    if nested.exists() and (nested / "model.safetensors").exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(local_ckpt / item.name))
        nested.rmdir()

    assert (local_ckpt / "model.safetensors").exists(), f"Download failed: {local_ckpt}"


def resolve_checkpoint_refs(ckpt_names: list[str], checkpoints: list[str]) -> list[str]:
    if "first" not in checkpoints and "last" not in checkpoints:
        return checkpoints

    ckpt_nums = []
    for name in ckpt_names:
        if name.startswith("checkpoint-"):
            num = name.replace("checkpoint-", "")
            if num.isdigit():
                ckpt_nums.append(int(num))

    assert ckpt_nums, "No checkpoints found"
    ckpt_nums.sort()
    first, last = str(ckpt_nums[0]), str(ckpt_nums[-1])

    resolved = []
    for ckpt in checkpoints:
        if ckpt == "first":
            resolved.append(first)
        elif ckpt == "last":
            resolved.append(last)
        else:
            resolved.append(ckpt)

    return list(dict.fromkeys(resolved))


def run_eval(
    run_name: str,
    cfg: EvalConfig,
    local_ckpt_dir: Path,
    evals_to_run: list[tuple[str, str, int, float]],
):
    script_dir = Path(__file__).parent

    for ckpt, dataset, seed, temp in evals_to_run:
        ckpt_dir = local_ckpt_dir / f"checkpoint-{ckpt}"
        if ckpt.startswith("baseline-"):
            policy_path = ckpt_dir / ".baseline_marker"
        else:
            policy_path = ckpt_dir / "model.safetensors"
            if not policy_path.exists():
                print(f"  Skipping {ckpt}: policy not found")
                continue

        output_dir = (
            Path(cfg.save_path)
            / run_name
            / f"checkpoint-{ckpt}_seed_{seed}_temp_policy_{temp}"
        )
        if cfg.sampling_mode:
            output_dir = Path(f"{output_dir}_sampling_mode_{cfg.sampling_mode}")
        if cfg.adaptive_block:
            # B0 belongs in the name: it is only the AdaBlock fallback length,
            # but it changes the realized block sizes, and without it two Ada
            # runs differing only in B0 write to the same directory.
            output_dir = Path(
                f"{output_dir}_ada{cfg.delimiter_threshold}"
                f"_B{resolve_block_length(cfg)}"
            )
        if cfg.remasking == "block_schedule":
            # Must not collide with the learned-policy dir for the same checkpoint;
            # the schedule itself is the only thing that distinguishes the runs.
            tag = cfg.block_schedule.replace(":", "").replace(",", "-")
            output_dir = Path(f"{output_dir}_sched{tag}")
        if cfg.remasking == "block_policy":
            # The dir name encodes neither block length nor threshold (the policy
            # picks both per block), so tag it to keep these runs separate from a
            # fixed-block eval of the same checkpoint.
            output_dir = Path(f"{output_dir}_blockpolicy")
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(torch.cuda.device_count() or 1),
            "-m",
            "eval.eval",
            "--config",
            cfg.config_path,
            "--dataset",
            dataset,
            "--batch_size",
            "1",
            "--remasking",
            cfg.remasking,
            "--policy_path",
            str(policy_path),
            "--output_dir",
            str(output_dir),
            "--model_path",
            cfg.model_path,
            "--seed",
            str(seed),
            "--temperature_policy",
            str(temp),
        ]
        if cfg.sampling_mode:
            cmd.extend(["--sampling_mode", cfg.sampling_mode])
        if cfg.block_length:
            cmd.extend(["--block_length", str(cfg.block_length)])
        if cfg.gen_length:
            cmd.extend(["--gen_length", str(cfg.gen_length)])
        if cfg.n_test:
            cmd.extend(["--n_test", str(cfg.n_test)])
        if cfg.adaptive_block:
            cmd.extend(
                ["--adaptive_block", "--delimiter_threshold", str(cfg.delimiter_threshold)]
            )
        if cfg.remasking == "block_schedule":
            cmd.extend(["--block_schedule", cfg.block_schedule])

        print(f"  Eval: {ckpt} seed={seed} temp={temp} {dataset}")
        result = subprocess.run(cmd, cwd=script_dir.parent)
        assert result.returncode == 0, f"Evaluation failed for {ckpt} {dataset}"


def aggregate(save_path: str):
    save_path = Path(save_path)
    if not save_path.exists():
        print(f"Warning: save_path {save_path} does not exist, skipping aggregation")
        return

    gen_files = list(save_path.glob("**/*_generations.json"))
    if not gen_files:
        print("No results found to aggregate")
        return

    cmd = ["python", "-m", "eval.aggregate_results", "--results_dir", str(save_path)]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print("Warning: Aggregation failed")


def run_pipeline(cfg: EvalConfig) -> str:
    run_name = Path(cfg.run_path).name
    is_local = not cfg.run_path.startswith("s3://")

    if is_local:
        local_ckpt_dir = Path(cfg.run_path)
    else:
        local_ckpt_dir = Path("temp_checkpoints") / run_name

    print(f"\n=== {run_name} ===")

    # For baselines, "self" checkpoint means use the run name as checkpoint
    if run_name.startswith("baseline-") and "self" in cfg.checkpoints:
        cfg = replace(
            cfg, checkpoints=[run_name if c == "self" else c for c in cfg.checkpoints]
        )

    if cfg.checkpoints and not run_name.startswith("baseline-"):
        if is_local:
            ckpt_names = [p.name for p in local_ckpt_dir.iterdir()]
        else:
            s3 = get_s3()
            ckpt_names = [
                Path(p).name for p in s3.ls(cfg.run_path.replace("s3://", ""))
            ]
        resolved = resolve_checkpoint_refs(ckpt_names, cfg.checkpoints)
        if resolved != cfg.checkpoints:
            print(f"Resolved checkpoints: {cfg.checkpoints} -> {resolved}")
            cfg = replace(cfg, checkpoints=resolved)

    all_runs = [
        (ckpt, ds, seed, temp)
        for ckpt in cfg.checkpoints
        for ds in cfg.datasets
        for seed in cfg.seeds
        for temp in cfg.temperatures
    ]

    print(f"Running {len(all_runs)} evaluations.")

    needed_ckpts = list(set(cfg.checkpoints))
    if not run_name.startswith("baseline-") and not is_local:
        s3 = get_s3()
        print(f"Downloading: {needed_ckpts}")
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for ckpt in needed_ckpts:
            download_checkpoint(s3, cfg.run_path, ckpt, local_ckpt_dir)

    run_eval(run_name, cfg, local_ckpt_dir, all_runs)
    return run_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_paths")
    parser.add_argument("config_path")
    parser.add_argument("--datasets", default="gsm8k")
    parser.add_argument("--temperatures", default="1.0")
    parser.add_argument("--sampling_mode", default=None)
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--save_path", default="./eval_results")
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--adaptive_block", action="store_true")
    parser.add_argument("--delimiter_threshold", type=float, default=0.3)
    parser.add_argument(
        "--remasking",
        default="policy",
        choices=["policy", "block_policy", "block_schedule"],
        help="Learned strategy to evaluate. Ignored for baseline-* run paths, "
        "whose method is parsed out of the checkpoint name.",
    )
    parser.add_argument(
        "--block_schedule",
        default=None,
        help="Fixed 'b:thres,...' action list for --remasking block_schedule.",
    )
    parser.add_argument("--no_aggregate", action="store_true")
    args = parser.parse_args()
    if args.remasking == "block_schedule" and not args.block_schedule:
        parser.error("--remasking block_schedule requires --block_schedule")

    run_paths = (
        args.run_paths.split(";")
        if ";" in args.run_paths
        else args.run_paths.split(",")
    )
    datasets = args.datasets.split(",")
    if datasets == ["all"]:
        datasets = ["gsm8k", "math", "humaneval", "mbpp"]

    print(f"Evaluating {len(run_paths)} run(s)")

    run_names = []
    failed_runs = []
    for run_path in run_paths:
        cfg = EvalConfig(
            run_path=run_path.strip(),
            config_path=args.config_path,
            datasets=datasets,
            temperatures=[float(t) for t in args.temperatures.split(",")],
            sampling_mode=args.sampling_mode,
            checkpoints=[c for c in args.checkpoints.split(",") if c],
            seeds=[int(s) for s in args.seeds.split(",")],
            block_length=args.block_length,
            gen_length=args.gen_length,
            model_path=args.model_path,
            save_path=args.save_path,
            n_test=args.n_test,
            adaptive_block=args.adaptive_block,
            delimiter_threshold=args.delimiter_threshold,
            remasking=args.remasking,
            block_schedule=args.block_schedule,
        )
        try:
            run_names.append(run_pipeline(cfg))
        except Exception as e:
            print(f"Warning: Run {Path(run_path).name} failed: {e}")
            failed_runs.append(Path(run_path).name)

    print(f"\nDone. Evaluated {len(run_names)} run(s).")
    if failed_runs:
        print(f"Failed runs: {failed_runs}")
        raise RuntimeError(
            f"Evaluation failed for {len(failed_runs)} run(s): "
            + ", ".join(failed_runs)
        )

    if not args.no_aggregate:
        print("\nAggregating results...")
        aggregate(args.save_path)


if __name__ == "__main__":
    main()
