#!/usr/bin/env python3
"""Plot the GSM8K accuracy/NFE Pareto frontier for selected evaluations."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SELECTED_RUNS = {
    ("baseline-low_confidence-K128", "baseline-low_confidence-K128"): "K128",
    ("baseline-low_confidence-K256", "baseline-low_confidence-K256"): "K256",
    ("baseline-fastdllm-t0.9", "baseline-fastdllm-t0.9"): "Fast-dLLM",
    ("llada8b_bl32", "1870"): "Last ckpt (1870)",
}


def load_points(summary_csv: Path) -> list[dict]:
    points = []
    with summary_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            label = SELECTED_RUNS.get((row["run"], row["checkpoint"]))
            if label is None or row["dataset"] != "gsm8k":
                continue
            points.append(
                {
                    "label": label,
                    "nfe": float(row["avg_steps_mean"]),
                    "accuracy": float(row["accuracy_mean"]),
                    "nfe_std": float(row["avg_steps_std"] or 0),
                    "accuracy_std": float(row["accuracy_std"] or 0),
                }
            )
    if len(points) != len(SELECTED_RUNS):
        found = {point["label"] for point in points}
        missing = set(SELECTED_RUNS.values()) - found
        raise ValueError(f"Missing evaluation rows: {sorted(missing)}")
    return points


def is_pareto_optimal(point: dict, points: list[dict]) -> bool:
    """NFE is minimized and accuracy is maximized."""
    return not any(
        other["nfe"] <= point["nfe"]
        and other["accuracy"] >= point["accuracy"]
        and (
            other["nfe"] < point["nfe"]
            or other["accuracy"] > point["accuracy"]
        )
        for other in points
        if other is not point
    )


def plot(points: list[dict], output: Path) -> None:
    frontier = sorted(
        (point for point in points if is_pareto_optimal(point, points)),
        key=lambda point: point["nfe"],
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.8))

    ax.plot(
        [point["nfe"] for point in frontier],
        [point["accuracy"] for point in frontier],
        color="#D1495B",
        linewidth=2.2,
        linestyle="--",
        label="Pareto frontier",
        zorder=2,
    )

    offsets = {
        "Last ckpt (1870)": (10, -2),
        "Fast-dLLM": (10, -28),
        "K128": (10, -2),
        "K256": (-105, -28),
    }
    colors = {
        "Last ckpt (1870)": "#2A9D8F",
        "Fast-dLLM": "#E76F51",
        "K128": "#7A7A7A",
        "K256": "#457B9D",
    }

    for point in points:
        pareto = is_pareto_optimal(point, points)
        ax.errorbar(
            point["nfe"],
            point["accuracy"],
            xerr=point["nfe_std"] or None,
            yerr=point["accuracy_std"] or None,
            fmt="o" if pareto else "X",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.2,
            color=colors[point["label"]],
            ecolor=colors[point["label"]],
            elinewidth=1.5,
            capsize=4,
            zorder=3,
        )
        annotation = (
            f'{point["label"]}\n'
            f'{point["accuracy"]:.2f}% accuracy, {point["nfe"]:.1f} NFE'
        )
        ax.annotate(
            annotation,
            (point["nfe"], point["accuracy"]),
            xytext=offsets[point["label"]],
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold" if pareto else "normal",
        )

    ax.set_title("GSM8K Accuracy vs. Inference Compute", fontsize=15, pad=12)
    ax.set_xlabel("Average NFE (lower is better)", fontsize=11)
    ax.set_ylabel("Accuracy (%) (higher is better)", fontsize=11)
    ax.set_xlim(60, 275)
    ax.set_ylim(73.5, 80.3)
    ax.legend(loc="lower right", frameon=True)
    ax.text(
        0.01,
        0.02,
        "Error bars: +/-1 std across 3 seeds for last ckpt; baselines use 1 seed.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(load_points(args.summary_csv), args.output)


if __name__ == "__main__":
    main()
