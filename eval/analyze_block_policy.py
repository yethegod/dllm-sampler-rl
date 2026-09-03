#!/usr/bin/env python3
"""Summarize what the joint (block size, threshold) policy actually chose.

`eval.aggregate_results` reports accuracy and NFE but collapses the per-block
actions, which are the whole point of a block_policy run. This reads the same
`*_generations.json` files and reports, per evaluation directory:

  - accuracy and mean NFE (the two headline numbers, recomputed here so the
    action breakdown and the score always come from one pass over one file)
  - the marginal distribution over block sizes and over thresholds
  - the joint (block size, threshold) distribution -- the marginals hide
    whether the policy couples the two heads
  - block size by block index, i.e. whether the schedule varies along the
    sequence or is stationary
  - accuracy and NFE split by the sequence's mean block size

Usage:
    python -m eval.analyze_block_policy <results_dir> [--csv out.csv]
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter
from collections import defaultdict

from common.parsing.parse_and_get_acc import check_gsm_correct
from common.parsing.parse_and_get_acc import extract_gsm_answer


def load_run(json_file):
    """Per-sample records for one evaluation, or None if it has no joint actions."""
    with open(json_file, "r") as f:
        data = json.load(f)

    samples = []
    for item in data.get("generations", []):
        blocks = item.get("block_sizes")
        thres = item.get("thresholds")
        # Only the learned block-size runs carry per-sample action lists. block_policy
        # populates both (index-aligned); block_unmask_policy has no thresholds, so
        # None is padded in and the threshold summaries below are skipped.
        if not blocks or item.get("action_logits") is None:
            continue
        if not thres:
            thres = [None] * len(blocks)
        answer = extract_gsm_answer(item.get("generations", ""))
        samples.append(
            {
                "correct": bool(check_gsm_correct(answer, item.get("ground_truth"))),
                "steps": item.get("steps", 0),
                "blocks": list(blocks),
                "thresholds": list(thres),
            }
        )
    return samples or None


def summarize(samples):
    """Accuracy, NFE and the action distributions for one evaluation."""
    n = len(samples)
    block_hist = Counter()
    thres_hist = Counter()
    joint_hist = Counter()
    by_index = defaultdict(Counter)

    for s in samples:
        for i, (b, t) in enumerate(zip(s["blocks"], s["thresholds"])):
            block_hist[b] += 1
            by_index[i][b] += 1
            if t is not None:
                thres_hist[round(t, 4)] += 1
                joint_hist[(b, round(t, 4))] += 1

    n_decisions = sum(block_hist.values())
    return {
        "n_samples": n,
        "accuracy": 100.0 * sum(s["correct"] for s in samples) / n,
        "nfe": sum(s["steps"] for s in samples) / n,
        "blocks_per_seq": n_decisions / n,
        "mean_block_size": sum(b * c for b, c in block_hist.items()) / n_decisions,
        "mean_threshold": (
            sum(t * c for t, c in thres_hist.items()) / n_decisions
            if thres_hist
            else None
        ),
        "block_hist": block_hist,
        "thres_hist": thres_hist,
        "joint_hist": joint_hist,
        "by_index": by_index,
        "n_decisions": n_decisions,
    }


def _pct(hist, total):
    return "  ".join(
        f"{k}: {100.0 * hist[k] / total:5.1f}%" for k in sorted(hist)
    )


def print_report(name, r):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(
        f"  samples {r['n_samples']}   accuracy {r['accuracy']:.2f}%   "
        f"NFE {r['nfe']:.1f}   blocks/seq {r['blocks_per_seq']:.2f}"
    )
    mean_t = r["mean_threshold"]
    print(
        f"  mean block size {r['mean_block_size']:.1f}   "
        + (f"mean threshold {mean_t:.3f}" if mean_t is not None else "no threshold axis")
    )

    nd = r["n_decisions"]
    print(f"\n  block size   ({nd} decisions)\n    {_pct(r['block_hist'], nd)}")
    if r["thres_hist"]:
        print(f"  threshold\n    {_pct(r['thres_hist'], nd)}")

        print("\n  threshold | block size   (row-normalized, blank = never chosen)")
        sizes = sorted(r["block_hist"])
        thresholds = sorted(r["thres_hist"])
        print("    " + "tau \\ b".ljust(10) + "".join(f"{b:>9}" for b in sizes))
        for t in thresholds:
            row = f"    {t:<10.2f}"
            for b in sizes:
                c = r["joint_hist"].get((b, t), 0)
                row += f"{100.0 * c / nd:>8.1f}%" if c else "        -"
            print(row)

    print("\n  block size by block index   (mean, count)")
    for i in sorted(r["by_index"]):
        h = r["by_index"][i]
        tot = sum(h.values())
        if tot < 0.02 * r["n_samples"]:  # tail indices only a few rows reach
            continue
        mean = sum(b * c for b, c in h.items()) / tot
        print(f"    block {i:<3} mean {mean:6.1f}   n={tot}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Directory of eval output directories")
    parser.add_argument("--csv", help="Also write the headline rows to this CSV")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "**", "*_generations.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"No *_generations.json under {args.results_dir}")

    rows = []
    for json_file in files:
        samples = load_run(json_file)
        if samples is None:
            continue  # not a block_policy run
        name = os.path.relpath(json_file, args.results_dir)
        r = summarize(samples)
        print_report(name, r)
        rows.append(
            {
                "run": name,
                "n_samples": r["n_samples"],
                "accuracy": round(r["accuracy"], 2),
                "nfe": round(r["nfe"], 1),
                "blocks_per_seq": round(r["blocks_per_seq"], 2),
                "mean_block_size": round(r["mean_block_size"], 1),
                "mean_threshold": (
                    round(r["mean_threshold"], 3)
                    if r["mean_threshold"] is not None
                    else None
                ),
                **{
                    f"frac_b{b}": round(100.0 * c / r["n_decisions"], 1)
                    for b, c in sorted(r["block_hist"].items())
                },
                **{
                    f"frac_t{t}": round(100.0 * c / r["n_decisions"], 1)
                    for t, c in sorted(r["thres_hist"].items())
                },
            }
        )

    if not rows:
        raise SystemExit(
            f"Found {len(files)} generations file(s) but none record per-block "
            "actions -- this analysis only applies to remasking=block_policy runs."
        )

    if args.csv:
        fields = sorted({k for row in rows for k in row}, key=lambda k: k != "run")
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
