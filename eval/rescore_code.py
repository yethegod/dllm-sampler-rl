"""Re-sanitize and re-score saved MBPP generations without decoding again.

Copies every *_generations.json under --src to the same relative path under --dst.
MBPP files get generation_sanitized recomputed from generation_raw with the current
data.sanitize.sanitize_mbpp and pass@1 re-run through HF code_eval; other files are
copied unchanged, so `eval.aggregate_results --results_dir <dst>` sees the full set.

Runs model-written code (HF code_eval): use a compute node, not a login node.

    HF_ALLOW_CODE_EVAL=1 python -m eval.rescore_code --src $WORK/eval_results/code \
        --dst $WORK/eval_results/code_rescored
"""

import argparse
import glob
import json
import os
import shutil

from data.sanitize import sanitize_mbpp
from eval.eval import evaluate_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(glob.escape(args.src), "**", "*_generations.json"), recursive=True))
    for f in files:
        out = os.path.join(args.dst, os.path.relpath(f, args.src))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.basename(f).startswith("mbpp"):
            shutil.copy2(f, out)
            continue
        with open(f) as fh:
            d = json.load(fh)
        gens = d["generations"]
        old = sum(g["pass@1"] for g in gens) / len(gens)
        n_changed = 0
        for g in gens:
            new = sanitize_mbpp(g["generation_raw"])
            n_changed += new.strip() != g["generation_sanitized"].strip()
            g["generation_sanitized"] = new
        d["code_eval_results"] = evaluate_code(gens, "mbpp")
        d["rescored_from"] = f
        with open(out, "w") as fh:
            json.dump(d, fh, indent=2)
        new_acc = sum(g["pass@1"] for g in gens) / len(gens)
        print(f"{os.path.relpath(f, args.src)}: {n_changed} re-sanitized, pass@1 {old:.3f} -> {new_acc:.3f}")


if __name__ == "__main__":
    main()
