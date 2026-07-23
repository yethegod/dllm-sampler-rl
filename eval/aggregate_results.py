#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import glob
import json
import os
import re

import pandas as pd

from common.parsing.parse_and_get_acc import parse_code_answers
from common.parsing.parse_and_get_acc import parse_gsm_answers
from common.parsing.parse_and_get_acc import parse_math_answers


def parse_checkpoint_name(path) -> int | str:
    basename = os.path.basename(path)

    match = re.search(r"checkpoint-(.+?)_seed_", basename)
    if match:
        checkpoint_id = match.group(1)
        try:
            return int(checkpoint_id)
        except ValueError:
            return checkpoint_id

    match = re.search(r"checkpoint-([^_]+)", basename)
    if match:
        checkpoint_id = match.group(1)
        try:
            return int(checkpoint_id)
        except ValueError:
            return checkpoint_id
    return 0


def parse_seed(path) -> int:
    match = re.search(r"seed_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else 42


def parse_temperature(path) -> float:
    match = re.search(r"temp_policy_([0-9.]+)", os.path.basename(path))
    return float(match.group(1)) if match else 1.0


def extract_dataset_name(json_file) -> str:
    filename = os.path.basename(json_file)
    match = re.match(r"^(gsm8k|math|humaneval|mbpp)_", filename)
    if match:
        return match.group(1)
    return "unknown"


def extract_run_name(json_file, results_dir):
    """Extract run name from path structure.

    Expected structure: {results_dir}/.../{run_name}/checkpoint-{N}_seed_{S}_.../{dataset}_generations.json
    Returns the directory component immediately before checkpoint-*.
    """
    rel_path = os.path.relpath(json_file, results_dir)
    parts = rel_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("checkpoint-"):
            if i > 0:
                return parts[i - 1]
            break
    return parts[0] if parts else "unknown"


def aggregate_results(results_dir):
    """Aggregate results from all evaluation runs (eval.py format only)."""

    # Find eval.py result files
    evalpy_files = glob.glob(
        os.path.join(glob.escape(results_dir), "**", "*_generations.json"),
        recursive=True,
    )

    if len(evalpy_files) == 0:
        print(f"No result files found in {results_dir}")
        print("  Searched for: *_generations.json (eval.py format)")
        return None

    print(f"Found {len(evalpy_files)} result files to aggregate")

    # Parse all results
    all_results = []

    for json_file in evalpy_files:
        try:
            checkpoint_dir = os.path.dirname(json_file)
            checkpoint_num = parse_checkpoint_name(checkpoint_dir)
            seed = parse_seed(checkpoint_dir)
            temperature = parse_temperature(checkpoint_dir)
            run_name = extract_run_name(json_file, results_dir)
            dataset_name = extract_dataset_name(json_file)

            with open(json_file, "r") as f:
                data = json.load(f)

            if dataset_name in ["humaneval", "mbpp"]:
                (
                    total_correct,
                    total_processed,
                    processed_items,
                    total_effective_tokens,
                    steps,
                    wall_times,
                ) = parse_code_answers(json_data=data)
            elif dataset_name == "math":
                (
                    total_correct,
                    total_processed,
                    processed_items,
                    total_effective_tokens,
                    steps,
                    wall_times,
                ) = parse_math_answers(json_data=data)
            else:  # gsm8k and any other datasets
                (
                    total_correct,
                    total_processed,
                    processed_items,
                    total_effective_tokens,
                    steps,
                    wall_times,
                ) = parse_gsm_answers(json_data=data)

            accuracy = (
                (total_correct / total_processed * 100) if total_processed > 0 else 0
            )
            avg_steps = sum(steps) / len(steps) if steps else 0
            avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0
            avg_effective_tokens = (
                total_effective_tokens / total_processed if total_processed > 0 else 0
            )

            print(
                f"\nProcessing: {run_name}, checkpoint {checkpoint_num}, seed {seed}, temp {temperature}, dataset {dataset_name}"
            )
            print(f"  Accuracy: {accuracy:.2f}%")
            if avg_steps > 0:
                print(f"  Avg NFEs: {avg_steps:.1f}")
            if avg_wall_time > 0:
                print(f"  Avg Wall Time: {avg_wall_time:.3f}s")
            if avg_effective_tokens > 0:
                print(f"  Avg Tokens: {avg_effective_tokens:.1f}")

            # Check test set verification status to make sure we ran the whole thing (no samples dropped)
            test_verification = data.get("test_set_verification", {})
            expected_size = test_verification.get("expected_dataset_size", None)
            actual_size = test_verification.get(
                "actual_samples_processed", total_processed
            )
            coverage_complete = test_verification.get("coverage_complete", None)

            if coverage_complete is True:
                print(f"  Test set complete: {actual_size}/{expected_size} samples")
            elif coverage_complete is False:
                print(
                    f"  WARNING: Test set incomplete: {actual_size}/{expected_size} samples"
                )

            # Create result dict
            result = {
                "run": run_name,
                "dataset": dataset_name,
                "checkpoint": checkpoint_num,
                "seed": seed,
                "temperature": temperature,
                "accuracy": accuracy,
                "total_correct": total_correct,
                "total_processed": total_processed,
                "avg_steps": avg_steps,
                "avg_wall_time": avg_wall_time,
                "avg_effective_tokens": avg_effective_tokens,
                "total_effective_tokens": total_effective_tokens,
                "wall_time": data.get("metrics", {}).get("wall_time", 0),
                "model_path": data.get("model_path", ""),
                "remasking": data.get("remasking", ""),
                "sampling_mode": data.get("sampling_mode", "bernoulli"),
                # str: adaptive-block runs record e.g. "ada0.3" here, and mixed
                # int/str values break sorted() during grouping
                "block_length": str(data.get("block_length", 32)),
                "gen_length": data.get("gen_length", 256),
                "expected_dataset_size": expected_size,
                "actual_samples_processed": actual_size,
                "test_set_complete": coverage_complete,
                "json_file": json_file,
            }

            all_results.append(result)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_results:
        print("No valid results found")
        return None

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)

    # Sort by dataset, checkpoint, temperature, and seed
    df = df.sort_values(["dataset", "checkpoint", "temperature", "seed"])

    # Print summary statistics
    print("\n" + "=" * 80)
    print("AGGREGATION SUMMARY")
    print("=" * 80)
    print(f"Total evaluations: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Checkpoints: {df['checkpoint'].nunique()}")
    print(f"Temperatures: {sorted(df['temperature'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(
        f"Unique configurations: {len(df.groupby(['dataset', 'checkpoint', 'temperature', 'seed']))}"
    )
    print("=" * 80)

    return df


def create_summary_tables(df, output_dir):
    """Create summary tables with statistics across seeds, grouped by run, dataset, block_length, checkpoint and temperature."""

    has_steps = df["avg_steps"].sum() > 0
    has_wall_time = df["avg_wall_time"].sum() > 0
    has_tokens = df["avg_effective_tokens"].sum() > 0

    group_cols = ["run", "dataset", "block_length", "checkpoint", "temperature"]

    agg_dict = {
        "accuracy": ["mean", "std", "min", "max"],
    }

    if has_steps:
        agg_dict["avg_steps"] = ["mean", "std", "min", "max"]
    if has_wall_time:
        agg_dict["avg_wall_time"] = ["mean", "std", "min", "max"]
    if has_tokens:
        agg_dict["avg_effective_tokens"] = ["mean", "std", "min", "max"]

    summary_stats = df.groupby(group_cols).agg(agg_dict).round(4)

    # Flatten column names
    summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]

    # Add number of seeds
    summary_stats["num_seeds"] = df.groupby(group_cols).size()

    # Save detailed results
    detailed_path = os.path.join(output_dir, "detailed_results.csv")
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    # Save summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary table to console - separate by dataset and block_length
    for dataset in sorted(df["dataset"].unique()):
        for bl in sorted(df[df["dataset"] == dataset]["block_length"].unique()):
            dataset_stats = summary_stats.xs(
                (dataset, bl), level=["dataset", "block_length"]
            )

            print("\n" + "=" * 100)
            print(f"RESULTS FOR DATASET: {dataset.upper()}, BLOCK_LENGTH: {bl}")
            print("=" * 100)

            display_df = pd.DataFrame()
            display_df["Run"] = [idx[0] for idx in dataset_stats.index]
            display_df["Checkpoint"] = [idx[1] for idx in dataset_stats.index]
            display_df["Temp"] = [f"{idx[2]:.2f}" for idx in dataset_stats.index]

            display_df["Accuracy"] = [
                f"{row['accuracy_mean']:.2f}% +/- {row['accuracy_std']:.2f}"
                for _, row in dataset_stats.iterrows()
            ]
            display_df["Seeds"] = [
                f"{int(row['num_seeds'])}" for _, row in dataset_stats.iterrows()
            ]

            if has_steps:
                display_df["NFEs"] = [
                    f"{row['avg_steps_mean']:.1f} +/- {row['avg_steps_std']:.1f}"
                    if "avg_steps_mean" in row.index and "avg_steps_std" in row.index
                    else ""
                    for _, row in dataset_stats.iterrows()
                ]
            if has_wall_time:
                display_df["Time(s)"] = [
                    f"{row['avg_wall_time_mean']:.3f} +/- {row['avg_wall_time_std']:.3f}"
                    if "avg_wall_time_mean" in row.index
                    and "avg_wall_time_std" in row.index
                    else ""
                    for _, row in dataset_stats.iterrows()
                ]
            if has_tokens:
                display_df["Tokens"] = [
                    f"{row['avg_effective_tokens_mean']:.1f} +/- {row['avg_effective_tokens_std']:.1f}"
                    if "avg_effective_tokens_mean" in row.index
                    and "avg_effective_tokens_std" in row.index
                    else ""
                    for _, row in dataset_stats.iterrows()
                ]

            # Print using pandas to_string with nice formatting
            print(display_df.to_string(index=False))
            print("=" * 100)

    # Create a human-readable report
    report_path = os.path.join(output_dir, "results_report.txt")
    with open(report_path, "w") as f:
        f.write("Evaluation Results Summary\n")
        f.write("=" * 50 + "\n\n")

        # Overall statistics
        f.write(f"Total evaluations: {len(df)}\n")
        f.write(
            f"Datasets evaluated: {df['dataset'].nunique()} {sorted(df['dataset'].unique())}\n"
        )
        f.write(f"Checkpoints evaluated: {df['checkpoint'].nunique()}\n")
        f.write(
            f"Temperatures evaluated: {df['temperature'].nunique()} {sorted(df['temperature'].unique())}\n"
        )
        f.write(
            f"Seeds per configuration: {df.groupby(['dataset', 'checkpoint', 'temperature']).size().iloc[0]}\n"
        )

        # Configuration (from first result)
        first_result = df.iloc[0]
        f.write(
            f"Configuration: BL={first_result['block_length']}, L={first_result['gen_length']}\n"
        )
        f.write(
            f"Remasking: {first_result['remasking']}, Sampling: {first_result['sampling_mode']}\n"
        )
        f.write("\n")

        # Best results per dataset
        f.write("Best Results by Dataset:\n")
        f.write("-" * 50 + "\n")
        for dataset in sorted(df["dataset"].unique()):
            dataset_df = df[df["dataset"] == dataset]
            best = dataset_df.loc[dataset_df["accuracy"].idxmax()]
            f.write(f"\n{dataset.upper()}:\n")
            f.write(f"  Checkpoint: {best['checkpoint']}\n")
            f.write(f"  Temperature: {best['temperature']:.2f}\n")
            f.write(f"  Seed: {best['seed']}\n")
            f.write(f"  Accuracy: {best['accuracy']:.2f}%\n")
            if has_steps and best["avg_steps"] > 0:
                f.write(f"  Avg Steps: {best['avg_steps']:.1f}\n")
            if has_wall_time and best["avg_wall_time"] > 0:
                f.write(f"  Avg Wall Time: {best['avg_wall_time']:.3f}s\n")

        f.write("\n\n")

        # Summary table per dataset and block_length
        for dataset in sorted(df["dataset"].unique()):
            for bl in sorted(df[df["dataset"] == dataset]["block_length"].unique()):
                dataset_stats = summary_stats.xs(
                    (dataset, bl), level=["dataset", "block_length"]
                )

                f.write(f"\nResults for {dataset.upper()}, BL={bl}:\n")
                f.write("-" * 95 + "\n")

                header = f"{'Run':>20} {'Checkpoint':>15} {'Temp':>6} {'Accuracy':>12} {'Std':>8} {'Seeds':>6}"
                if has_steps:
                    header += f" {'NFEs':>15}"
                if has_wall_time:
                    header += f" {'Time(s)':>15}"
                if has_tokens:
                    header += f" {'Tokens':>15}"
                f.write(header + "\n")
                f.write("-" * 95 + "\n")

                for idx in sorted(
                    dataset_stats.index,
                    key=lambda x: (x[0], isinstance(x[1], str), x[1], x[2]),
                ):
                    run, checkpoint, temp = idx
                    row = dataset_stats.loc[idx]
                    line = f"{run:>20} {checkpoint:>15} {temp:>6.2f} {row['accuracy_mean']:>8.2f}% +/- {row['accuracy_std']:>4.2f} {int(row['num_seeds']):>6}"
                    if has_steps and "avg_steps_mean" in row.index:
                        line += f" {row['avg_steps_mean']:>6.1f} +/- {row['avg_steps_std']:>4.1f}"
                    if has_wall_time and "avg_wall_time_mean" in row.index:
                        line += f" {row['avg_wall_time_mean']:>6.3f} +/- {row['avg_wall_time_std']:>4.3f}"
                    if has_tokens and "avg_effective_tokens_mean" in row.index:
                        line += f" {row['avg_effective_tokens_mean']:>6.1f} +/- {row['avg_effective_tokens_std']:>4.1f}"
                    f.write(line + "\n")

    print(f"Results report saved to: {report_path}")

    return summary_stats


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--results_dir",
        action="append",
        required=True,
        help="Directory containing evaluation results (can specify multiple)",
    )
    parser.add_argument(
        "--output_dir", help="Output directory (defaults to first results_dir)"
    )

    args = parser.parse_args()

    results_dirs = args.results_dir
    output_dir = args.output_dir or results_dirs[0]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if len(results_dirs) == 1:
        print(f"Aggregating results from: {results_dirs[0]}")
        print(f"Output directory: {output_dir}")

        df = aggregate_results(results_dirs[0])

        if df is None:
            print("No results to aggregate")
            return

        create_summary_tables(df, output_dir)

        print("\nAggregation complete!")
        print(
            f"Processed {len(df)} evaluations from {df['run'].nunique()} runs, {df['checkpoint'].nunique()} checkpoints across {df['dataset'].nunique()} datasets"
        )
        print(f"Results saved to: {output_dir}")
    else:
        # Multiple runs - add run column and concatenate
        print(f"Aggregating results from {len(results_dirs)} runs:")
        for rd in results_dirs:
            print(f"  - {rd}")
        print(f"Output directory: {output_dir}")

        all_dfs = []
        for results_dir in results_dirs:
            print(f"\nProcessing: {results_dir}")

            df = aggregate_results(results_dir)
            if df is not None:
                all_dfs.append(df)
                print(f"  Found {len(df)} evaluations")
            else:
                print("  No results found")

        if not all_dfs:
            print("\nNo results to aggregate")
            return

        # Concatenate all dataframes
        df = pd.concat(all_dfs, ignore_index=True)

        print(f"\nTotal: {len(df)} evaluations from {df['run'].nunique()} runs")
        print(f"Runs: {sorted(df['run'].unique())}")

        # Create summary tables (will handle 'run' column)
        create_summary_tables(df, output_dir)

        print("\nAggregation complete!")
        print(
            f"Processed {len(df)} evaluations from {df['run'].nunique()} runs, {df['checkpoint'].nunique()} checkpoints across {df['dataset'].nunique()} datasets"
        )
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
