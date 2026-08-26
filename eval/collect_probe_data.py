"""Collect the Stage 0 probe dataset: does the hidden state carry signal beyond top-k?

The question this answers, before any policy is trained on hidden states: is the
observation the bottleneck at all? Every policy in this repo sees one number per
position (the top-1 confidence). Feeding it the dLLM's 4096-d hidden state instead
costs a projector, a replay path, and days of GPU -- worth it only if that state
separates good unmasking decisions better than the confidences already do.

The label is free and dense. `generate_unified(record_probe_data=True)` records, per
decoding step, the token state the decision was made from, the dLLM's argmax x0 at
every position, and the set of positions the decision actually ranged over. Comparing
x0 against the *final* sequence gives

    label(t, i) = [ x0_t[i] == final[i] ]   for positions actionable at step t

read as "had we unmasked position i at step t, would it have committed the token that
eventually won?" -- 1 means safe to commit now, 0 means wait. Unlike the confidences
this uses future information, so it is a privileged teacher rather than a restatement
of the input.

ROW SUPPORT (schema 2, and the reason this schema exists). A row is kept only if the
position was **actionable**: still masked *and inside the block being decoded at that
step*, i.e. a position the decoder was allowed to pick. Schema 1 kept every masked
position, which is mostly the not-yet-reached tail of the sequence -- positions no
decoding rule could have touched, whose x0 is an unconditioned guess. Those rows are
easy negatives that inflate every probe's AUC by the same amount and answer a question
the policy is never asked. Rows are additionally restricted to positions that are
*decided* in the final sequence: a position the decoder never got to is still mask_id
at the end, so its label would degenerate to (x0 == mask_id) = always 0 -- "ran out of
steps", not "unsafe".

Decoding uses a FIXED baseline -- Fast-dLLM at --thres, or `random` at --steps -- never
a policy under training, so the label does not depend on the thing being evaluated.
Only those two fixed-block modes are supported; adaptive-block and the per-row
block_policy / block_schedule loops have no single actionable set to record and are
rejected up front.

Features are stored as the *normalised* hidden state -- LayerNorm without affine, which
is exactly what HiddenProjInputMixin applies before its projection -- so the probe sees
the representation the policy would actually get, and the values stay in fp16 range
(raw LLaDA hidden states have outlier channels that do not).

Every invocation collects fresh data. There is no reuse path: the row support changed
underneath these filenames once already, and refitting a stale file is exactly the
failure this schema exists to prevent. An existing file at --out is not overwritten
either -- it is moved to a timestamped `.bak.npz` sibling first, then collection runs.

Example:
    python -m eval.collect_probe_data \
        --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --dataset gsm8k --n_test 384 --remasking fastdllm --thres 0.9 \
        --out /work/hdd/bhta/zsun9/probe_data/gsm8k_t09_actionable_fastdllm.npz

    python -m eval.collect_probe_data ... --remasking random --steps 256 \
        --out /work/hdd/bhta/zsun9/probe_data/gsm8k_random_k256_actionable.npz

This module keeps its heavy imports (torch / transformers / trl) inside main() so the
schema constants and the NPZ helpers below can be imported -- by eval.fit_probes and by
the tests -- without pulling in the model stack.
"""

import argparse
import os
import re
import subprocess
import time

import numpy as np

TOP_K = 16

# Bumped whenever the meaning of a row changes. Schema 1 (unversioned, no provenance)
# scored every masked position; schema 2 scores only actionable ones, so the two are
# not comparable and must never be pooled or silently reused.
PROBE_SCHEMA_VERSION = 2

# What one row *is*. Stored alongside the version so a future support change that keeps
# the same columns still trips the check.
PROBE_ROW_SUPPORT = "actionable_current_block_and_decided"

# Supported decoding baselines. Both are fixed-block and policy-free.
SUPPORTED_REMASKING = ("fastdllm", "random")

# Sentinels for the provenance field that does not apply to the chosen baseline.
# Stored rather than omitted so every file has the same key set.
THRESHOLD_NA = -1.0
STEPS_NA = -1

# Both example counts are recorded because they answer different questions:
# `n_examples` is how many examples the collector actually ran (it can stop early on
# --max_rows, which is what makes "did we really get 384?" checkable), while
# `n_examples_with_rows` is how many of those contributed at least one row.

# Recognised values of the tracked-working-tree flag, and the placeholder used when
# git could not be interrogated at all.
GIT_STATUSES = ("clean", "dirty", "unknown")
GIT_COMMIT_UNKNOWN = "unknown"

# Every provenance key a schema-2 file must carry.
PROVENANCE_KEYS = (
    "schema_version",
    "row_support",
    "remasking",
    "threshold",
    "decoding_steps",
    "seed",
    "dataset",
    "config",
    "n_examples_requested",
    "n_examples",
    "n_examples_with_rows",
    "batch_size",
    "gen_length",
    "block_length",
    "step_stride",
    "rows_per_step",
    "max_rows",
    "top_k",
    "n_candidates",
    "n_rows",
    "git_commit",
    "git_tracked_clean",
)

DATA_KEYS = ("h", "conf", "label", "example_id", "step", "pos")


# --------------------------------------------------------------------------------
# NPZ contract: provenance, validation, loading
# --------------------------------------------------------------------------------


def _unwrap(value):
    """np.savez stores scalars as 0-d arrays; give back the Python value."""
    arr = np.asarray(value)
    return arr.item() if arr.ndim == 0 else arr


def read_provenance(data) -> dict:
    """Pull the provenance keys present in an opened NPZ into a plain dict."""
    return {k: _unwrap(data[k]) for k in PROVENANCE_KEYS if k in data.files}


def _require(path: str, ok, message: str) -> None:
    """Raise a ValueError naming the file unless `ok`."""
    if not ok:
        raise ValueError(f"{path}: {message}")


def _check_row_arrays(data, prov: dict, path: str) -> None:
    """Every row array describes the same n_rows rows, in the advertised shape.

    Each array is pulled out of the NPZ exactly once: `h` is the big one (rows x 4096),
    and indexing an NpzFile re-decompresses on every access.
    """
    n_rows = int(prov["n_rows"])
    top_k = int(prov["top_k"])
    _require(path, n_rows >= 1, f"n_rows is {n_rows}; an empty probe file is useless")
    _require(path, top_k >= 1, f"top_k is {top_k}")
    for key in DATA_KEYS:
        arr = data[key]
        want_rank = 2 if key in ("h", "conf") else 1
        _require(
            path,
            arr.ndim == want_rank,
            f"{key} has rank {arr.ndim}, expected {want_rank}"
            + (" (rows x features)" if want_rank == 2 else ""),
        )
        _require(
            path,
            arr.shape[0] == n_rows,
            f"{key} has {arr.shape[0]} rows but n_rows is {n_rows}; the row arrays "
            f"disagree",
        )
        if key == "conf":
            _require(
                path,
                arr.shape[1] == top_k,
                f"conf is {arr.shape[1]} wide but top_k is {top_k}",
            )
        if key == "h":
            _require(path, arr.shape[1] >= 1, "h has no feature columns")
    n_candidates = int(prov["n_candidates"])
    _require(
        path,
        n_candidates >= n_rows,
        f"n_candidates {n_candidates} < n_rows {n_rows}; rows cannot outnumber the "
        f"population they were subsampled from",
    )


def _check_example_counts(data, prov: dict, path: str) -> None:
    """The two example counts bracket each other and match the ids actually stored."""
    requested = int(prov["n_examples_requested"])
    n_examples = int(prov["n_examples"])
    with_rows = int(prov["n_examples_with_rows"])
    _require(path, requested >= 1, f"n_examples_requested is {requested}")
    _require(
        path,
        1 <= n_examples <= requested,
        f"n_examples {n_examples} outside 1..{requested} (n_examples_requested); "
        f"n_examples counts examples actually processed",
    )
    _require(
        path,
        1 <= with_rows <= n_examples,
        f"n_examples_with_rows {with_rows} outside 1..{n_examples} (n_examples); "
        f"only processed examples can contribute rows",
    )
    uniq = np.unique(data["example_id"])
    _require(
        path,
        uniq.size == with_rows,
        f"example_id holds {uniq.size} distinct examples but n_examples_with_rows is "
        f"{with_rows}",
    )
    _require(
        path,
        int(uniq.min()) >= 0 and int(uniq.max()) < n_examples,
        f"example_id spans {int(uniq.min())}..{int(uniq.max())}, outside "
        f"0..{n_examples - 1}",
    )


def _check_baseline(prov: dict, path: str) -> None:
    """The recorded baseline is one we support, with its own hyperparameter set.

    The sentinels are the point: a file that claims `random` while carrying a real
    threshold was collected by code that disagrees with this module about what produced
    the trajectories, and its labels cannot be trusted to be policy-free.
    """
    remasking = str(prov["remasking"])
    _require(
        path,
        remasking in SUPPORTED_REMASKING,
        f"remasking {remasking!r} is not one of {list(SUPPORTED_REMASKING)}",
    )
    threshold = float(prov["threshold"])
    steps = int(prov["decoding_steps"])
    if remasking == "fastdllm":
        _require(
            path,
            0.0 < threshold <= 1.0,
            f"fastdllm file needs a threshold in (0, 1], got {threshold}",
        )
        _require(
            path,
            steps == STEPS_NA,
            f"fastdllm file must store decoding_steps={STEPS_NA} (not applicable), "
            f"got {steps}",
        )
    else:  # random
        _require(path, steps >= 1, f"random file needs decoding_steps >= 1, got {steps}")
        _require(
            path,
            threshold == THRESHOLD_NA,
            f"random file must store threshold={THRESHOLD_NA} (not applicable), "
            f"got {threshold}",
        )


def _check_git(prov: dict, path: str) -> None:
    status = str(prov["git_tracked_clean"])
    _require(
        path,
        status in GIT_STATUSES,
        f"git_tracked_clean {status!r} is not one of {list(GIT_STATUSES)}",
    )
    commit = str(prov["git_commit"])
    _require(
        path,
        commit == GIT_COMMIT_UNKNOWN or re.fullmatch(r"[0-9a-f]{40}", commit),
        f"git_commit {commit!r} is neither a full 40-hex sha nor "
        f"{GIT_COMMIT_UNKNOWN!r}",
    )


def validate_probe_npz(data, path: str = "<npz>") -> dict:
    """Raise unless `data` is a probe file this code understands. Returns provenance.

    Two layers. The first refuses data collected under different rules, all with the
    same remedy (recollect): a legacy file with no version at all, a version that is not
    this one, a row support that is not the one the analysis assumes, or a missing key.
    The second refuses a file that contradicts *itself* -- row arrays of different
    lengths, a conf width that is not top_k, more rows than candidates, example counts
    that cannot both be true, a baseline carrying the other baseline's hyperparameter.
    Those cannot be fixed by recollecting the same way; they mean the writer and this
    contract have drifted, and fitting such a file silently reports numbers for a
    dataset nobody described.
    """
    if "schema_version" not in data.files:
        raise ValueError(
            f"{path}: legacy probe file with no schema_version. It was collected over "
            f"ALL masked positions, not actionable ones, so its AUCs are not comparable "
            f"with schema {PROBE_SCHEMA_VERSION}. Recollect with eval.collect_probe_data."
        )
    version = _unwrap(data["schema_version"])
    if int(version) != PROBE_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: probe schema version {int(version)}, expected "
            f"{PROBE_SCHEMA_VERSION}. Recollect with eval.collect_probe_data."
        )
    support = str(_unwrap(data["row_support"])) if "row_support" in data.files else None
    if support != PROBE_ROW_SUPPORT:
        raise ValueError(
            f"{path}: row support {support!r}, expected {PROBE_ROW_SUPPORT!r}. "
            f"Recollect with eval.collect_probe_data."
        )
    missing = [k for k in PROVENANCE_KEYS if k not in data.files]
    if missing:
        raise ValueError(f"{path}: provenance keys missing: {', '.join(missing)}")
    missing = [k for k in DATA_KEYS if k not in data.files]
    if missing:
        raise ValueError(f"{path}: data keys missing: {', '.join(missing)}")
    prov = read_provenance(data)
    _check_row_arrays(data, prov, path)
    _check_example_counts(data, prov, path)
    _check_baseline(prov, path)
    _check_git(prov, path)
    return prov


def load_probe_npz(path: str):
    """Open a probe NPZ, validate the contract, return (data, provenance).

    The handle is closed on rejection: the caller gets an exception, not a return value,
    so nothing else can close it and the open file would leak for the process's life.
    """
    data = np.load(path, allow_pickle=False)
    try:
        return data, validate_probe_npz(data, path)
    except BaseException:
        data.close()
        raise


def format_provenance(prov: dict) -> str:
    """One line per provenance field, for the top of any run that consumes the data."""
    width = max(len(k) for k in prov) if prov else 0
    return "\n".join(f"  {k:<{width}} {prov[k]}" for k in PROVENANCE_KEYS if k in prov)


def git_provenance(repo_dir: str) -> tuple[str, str]:
    """(full commit sha, 'clean' | 'dirty' | 'unknown') for the tracked working tree.

    Deliberately not a content hash of the source files: the commit plus the tracked
    diff status is what actually reproduces a run, and hashing files invites the
    illusion that an untracked edit was captured.
    """

    def _run(cmd):
        return subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout

    try:
        commit = _run(["git", "rev-parse", "HEAD"]).strip()
    except Exception:
        return "unknown", "unknown"
    try:
        dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"]).strip()
    except Exception:
        return commit, "unknown"
    return commit, "dirty" if dirty else "clean"


# --------------------------------------------------------------------------------
# Row support
# --------------------------------------------------------------------------------


def select_probe_rows(actionable_t, state_t, x0_t, final_gen, mask_id: int):
    """One step's (B, L) tensors -> (selected, label), both (B, L) bool.

    A row survives iff it is BOTH
      * actionable at this step -- masked and inside the block being decoded, i.e. a
        position this step's decision was allowed to pick; and
      * decided in the final sequence -- otherwise the label degenerates to
        (x0 == mask_id) = always 0, which records "ran out of steps", not "unsafe".

    The subset assertion is the cheap guard on the whole contract: an actionable
    position is masked by construction, so if that ever fails the recording hook and
    the decode loop have drifted apart and every label downstream is quietly wrong.
    """
    masked = state_t == mask_id
    assert bool(
        (actionable_t & ~masked).sum() == 0
    ), "actionable positions must be a subset of the masked state"
    selected = actionable_t & (final_gen != mask_id)
    return selected, (x0_t == final_gen) & selected


# --------------------------------------------------------------------------------
# Argument validation (runs before the 8B model is touched)
# --------------------------------------------------------------------------------


def validate_args(args) -> None:
    """Reject every unsupported collection setting, cheaply and with a reason."""
    if args.remasking not in SUPPORTED_REMASKING:
        raise ValueError(
            f"--remasking {args.remasking!r} is not supported; probe collection "
            f"supports exactly {list(SUPPORTED_REMASKING)} (fixed-block, policy-free)."
        )
    if args.remasking == "fastdllm" and args.thres is None:
        raise ValueError("--remasking fastdllm requires --thres")
    if args.remasking == "random" and args.steps is None:
        raise ValueError("--remasking random requires --steps")
    if args.remasking != "fastdllm" and args.thres is not None:
        raise ValueError(f"--thres is meaningless for --remasking {args.remasking!r}")
    if args.remasking != "random" and args.steps is not None:
        raise ValueError(f"--steps is meaningless for --remasking {args.remasking!r}")
    # --steps 0 reaches generate_unified as a division by zero, and a negative one
    # silently decodes nothing; the gen_length ceiling is checked later, once the
    # config has been read.
    if args.steps is not None and args.steps < 1:
        raise ValueError(f"--steps must be >= 1, got {args.steps}")
    if args.step_stride < 1:
        raise ValueError("--step_stride must be >= 1")
    if args.rows_per_step < 1:
        raise ValueError("--rows_per_step must be >= 1")
    if args.n_test < 1:
        raise ValueError(f"--n_test must be >= 1, got {args.n_test}")
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.max_rows < 1:
        raise ValueError(f"--max_rows must be >= 1, got {args.max_rows}")


def validate_config(cfg) -> None:
    """Reject configs whose decoding loop has no single actionable set to record."""
    if getattr(cfg, "adaptive_block", False):
        raise ValueError(
            "probe collection requires fixed-block decoding, but the config sets "
            "adaptive_block=True. Block boundaries chosen at decode time make the "
            "actionable set depend on the run being recorded."
        )
    cfg_remasking = getattr(cfg, "remasking", None)
    if cfg_remasking in ("block_policy", "block_schedule"):
        raise ValueError(
            f"config remasking={cfg_remasking!r} is a per-ROW block loop, which has no "
            f"single current block to record an actionable set against. Use a "
            f"fixed-block config; the decoding baseline comes from --remasking."
        )


# --------------------------------------------------------------------------------
# Output path: never reuse, never clobber
# --------------------------------------------------------------------------------


def resolve_out_path(out: str) -> str:
    """The path np.savez will actually write (it appends .npz when absent)."""
    return out if out.endswith(".npz") else out + ".npz"


def backup_existing(path: str) -> str | None:
    """Move an existing collection out of the way; return where it went, or None.

    There is no reuse mode, so every run recollects -- which means every run would
    otherwise overwrite whatever is already at --out, and an old file is someone's
    result. Timestamped rather than numbered so the log line says when it was taken.
    """
    if not os.path.exists(path):
        return None
    root = path[: -len(".npz")] if path.endswith(".npz") else path
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{root}.{stamp}.bak.npz"
    n = 0
    while os.path.exists(backup):  # same-second reruns
        n += 1
        backup = f"{root}.{stamp}-{n}.bak.npz"
    os.replace(path, backup)
    return backup


# --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_test", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--remasking",
        type=str,
        default="fastdllm",
        choices=list(SUPPORTED_REMASKING),
        help="Fixed decoding baseline that produces the trajectories. Both are "
        "policy-free, so the label cannot depend on the thing being evaluated.",
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=None,
        help="Fast-dLLM confidence threshold. Required by --remasking fastdllm, "
        "rejected otherwise.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total decoding steps. Required by --remasking random, rejected "
        "otherwise. 256 at gen_length 256 is one position per step.",
    )
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot", type=int, default=-1)
    parser.add_argument(
        "--rows_per_step",
        type=int,
        default=192,
        help="Actionable positions sampled per (batch, decoding step). Uniform within "
        "the step, so late steps -- which have few candidates left -- contribute "
        "everything they have and early steps are subsampled.",
    )
    parser.add_argument(
        "--step_stride",
        type=int,
        default=1,
        help="Replay only every k-th recorded step. Decoding still costs T forwards but "
        "the replay pass drops to T/k, which is what buys the example count that "
        "actually determines the probe's effective sample size. Adjacent steps differ "
        "by a handful of tokens anyway.",
    )
    parser.add_argument("--max_rows", type=int, default=400_000)
    parser.add_argument("--out", type=str, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    # Everything cheap and refusable happens before the 8B checkpoint is loaded.
    validate_args(args)
    out_path = resolve_out_path(args.out)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from transformers import AutoTokenizer
    from trl import TrlParser

    from common.config import Config
    from common.generation.generation import generate_unified
    from eval.eval import DATASET_MAP
    from eval.eval import FEW_SHOT_DEFAULTS
    from eval.eval import MASK_TOKENS_MAP
    from eval.eval import init_seed

    init_seed(args.seed)
    (cfg,) = TrlParser((Config,)).parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    validate_config(cfg)

    gen_length = args.gen_length or cfg.max_completion_length
    block_length = args.block_length or cfg.block_length
    # generate_unified asserts this deep inside the loop; catching it here keeps a
    # typo'd --steps from costing a model load first.
    if args.remasking == "random" and args.steps > gen_length:
        raise ValueError(
            f"--steps {args.steps} exceeds gen_length {gen_length}; at most one "
            f"position can be unmasked per step"
        )
    few_shot = FEW_SHOT_DEFAULTS[args.dataset] if args.few_shot == -1 else args.few_shot

    # Last refusable step before the 8B load, and the only place the output is touched:
    # there is no reuse path, so an existing file is moved aside now rather than
    # overwritten at the end.
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    backup = backup_existing(out_path)
    if backup:
        print(f"moved existing {out_path} -> {backup}", flush=True)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_commit, git_clean = git_provenance(repo_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        cfg.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    is_llada = "LLaDA" in cfg.model_path
    mask_id = MASK_TOKENS_MAP["LLaDA" if is_llada else "Dream"]
    model_type = "LLaDA" if is_llada else "Dream"
    assert is_llada, "hidden-state features are only wired up for LLaDA"

    dataset = DATASET_MAP[args.dataset](
        tokenizer=tokenizer, subsample=-1, num_examples=few_shot, add_reasoning=True
    )
    collate_fn = dataset.collate_fn
    if args.n_test < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    d_model = model.config.hidden_size
    feats_h, feats_c, labels, ex_ids, steps, poss = [], [], [], [], [], []
    n_rows = 0
    # Candidates BEFORE --rows_per_step subsampling: the size of the population the
    # dataset is a sample of. Without it a small file is ambiguous between "the
    # baseline offered few choices" and "we threw most of them away".
    n_candidates = 0
    example_offset = 0
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if n_rows >= args.max_rows:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].bool().to(device)
            P = input_ids.shape[1]

            result = generate_unified(
                model=model,
                prompt=input_ids,
                remasking=args.remasking,
                thres=args.thres,
                steps=args.steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                mask_id=mask_id,
                model_type=model_type,
                attention_mask=attn,
                record_probe_data=True,
            )
            final_gen = result.sequences[:, P:]  # (B, L)
            state_hist = result.state_history  # (B, T, L)
            x0_hist = result.x0_history  # (B, T, L)
            act_hist = result.actionable_history  # (B, T, L)
            assert act_hist is not None, (
                "actionable_history is required for schema "
                f"{PROBE_SCHEMA_VERSION}; generation.py is out of date"
            )
            # Time alignment is the whole contract: row (b, t, i) of all three must
            # describe the same pre-decision moment.
            assert state_hist.shape == x0_hist.shape == act_hist.shape, (
                f"probe histories disagree: state {tuple(state_hist.shape)}, "
                f"x0 {tuple(x0_hist.shape)}, actionable {tuple(act_hist.shape)}"
            )
            B, T, L = state_hist.shape

            # Replay each recorded step to get the features the policy would have seen.
            # This is the same reconstruction compute_loss uses, and it reproduces the
            # rollout's hidden states exactly because the dLLM is deterministic.
            attn_full = torch.ones((B, P + L), dtype=torch.float, device=device)
            attn_full[:, :P] = attn.float()
            attn_full = attn_full.to(model.dtype)

            for t in range(0, T, args.step_stride):
                sel, lab = select_probe_rows(
                    act_hist[:, t], state_hist[:, t], x0_hist[:, t], final_gen, mask_id
                )
                if not sel.any():
                    continue
                x_t = torch.cat([input_ids, state_hist[:, t]], dim=1)
                out = model(
                    x_t, attention_mask=attn_full, output_hidden_states=True
                )
                h = out.hidden_states[-1][:, P:, :]  # (B, L, d)
                probs = F.softmax(out.logits[:, P:], dim=-1)
                conf = probs.topk(TOP_K, dim=-1).values  # (B, L, K)
                # Same normalisation HiddenProjInputMixin applies before projecting.
                h = F.layer_norm(h.float(), (d_model,))

                bidx, pidx = sel.nonzero(as_tuple=True)
                n_candidates += int(bidx.numel())
                take = np.arange(bidx.numel())
                if bidx.numel() > args.rows_per_step:
                    take = rng.choice(bidx.numel(), args.rows_per_step, replace=False)
                take = torch.as_tensor(take, device=device)
                bsel, psel = bidx[take], pidx[take]

                feats_h.append(h[bsel, psel].to(torch.float16).cpu().numpy())
                feats_c.append(conf[bsel, psel].to(torch.float16).cpu().numpy())
                labels.append(lab[bsel, psel].to(torch.uint8).cpu().numpy())
                ex_ids.append((bsel + example_offset).cpu().numpy().astype(np.int32))
                steps.append(np.full(bsel.numel(), t, dtype=np.int16))
                poss.append(psel.cpu().numpy().astype(np.int16))
                n_rows += bsel.numel()

            example_offset += B
            print(
                f"batch {bi}: T={T} examples {example_offset} "
                f"candidates {n_candidates} rows so far {n_rows}",
                flush=True,
            )

    if not labels:
        raise RuntimeError("no actionable rows collected; nothing to write")

    out = dict(
        h=np.concatenate(feats_h),
        conf=np.concatenate(feats_c),
        label=np.concatenate(labels),
        example_id=np.concatenate(ex_ids),
        step=np.concatenate(steps),
        pos=np.concatenate(poss),
    )
    out.update(
        schema_version=np.int32(PROBE_SCHEMA_VERSION),
        row_support=PROBE_ROW_SUPPORT,
        remasking=args.remasking,
        threshold=np.float64(args.thres if args.thres is not None else THRESHOLD_NA),
        decoding_steps=np.int32(args.steps if args.steps is not None else STEPS_NA),
        seed=np.int32(args.seed),
        dataset=args.dataset,
        config=args.config,
        n_examples_requested=np.int32(args.n_test),
        # Examples the loop actually ran, not examples that happened to yield a row --
        # otherwise "did we get the 384 we asked for?" is unanswerable from the file,
        # and an early --max_rows stop looks the same as a baseline that offered few
        # actionable positions.
        n_examples=np.int32(example_offset),
        n_examples_with_rows=np.int32(len(np.unique(out["example_id"]))),
        batch_size=np.int32(args.batch_size),
        gen_length=np.int32(gen_length),
        block_length=np.int32(block_length),
        step_stride=np.int32(args.step_stride),
        rows_per_step=np.int32(args.rows_per_step),
        max_rows=np.int32(args.max_rows),
        top_k=np.int32(TOP_K),
        n_candidates=np.int64(n_candidates),
        n_rows=np.int64(len(out["label"])),
        git_commit=git_commit,
        git_tracked_clean=git_clean,
    )
    np.savez(out_path, **out)
    pos_rate = float(out["label"].mean())
    print(
        f"wrote {out_path}: {len(out['label'])} rows of {n_candidates} candidates, "
        f"h={out['h'].shape}, conf={out['conf'].shape}, "
        f"positive rate {pos_rate:.4f}, {int(out['n_examples'])} examples processed "
        f"({int(out['n_examples_with_rows'])} with rows) of {args.n_test} requested"
    )
    if example_offset < args.n_test:
        print(
            f"WARNING: only {example_offset} of {args.n_test} examples were processed "
            f"(--max_rows {args.max_rows} reached, or the dataset is smaller); the "
            f"probe's effective sample size is examples, not rows",
            flush=True,
        )
    with np.load(out_path, allow_pickle=False) as check:
        print("provenance:")
        print(format_provenance(validate_probe_npz(check, out_path)))


if __name__ == "__main__":
    main()
