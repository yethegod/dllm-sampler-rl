"""Tests for the actionable-position probe dataset (schema 2).

The Stage 0 probe used to score every masked position at every recorded step. Most of
those positions are the not-yet-reached tail of the sequence: no decoding rule was
allowed to pick them, their x0 is an unconditioned guess, and they are easy negatives
that lift every probe's AUC by the same amount. The dataset now keeps only *actionable*
rows -- masked AND inside the block being decoded -- and carries a version plus full
provenance so the two generations of data can never be pooled or refit by accident.

What is pinned here:

  1. `actionable_history` is time-aligned with `state_history` / `x0_history`: same
     shape, and the tokens that appear between consecutive states are exactly the
     positions that were actionable, carrying exactly the x0 recorded at that step;
  2. actionable is confined to the block being decoded, and follows the block cursor;
  3. actionable is always a subset of the masked state, in both supported baselines;
  4. the collector keeps rows that are actionable AND decided, and its subset assertion
     fires when the two drift apart;
  5. every unsupported mode is refused -- adaptive_block and the per-row block loops in
     `generate_unified`, and adaptive / block-policy / missing-hyperparameter settings
     in the collector, the latter before any model is loaded;
  6. a written NPZ carries the full provenance contract and validates, and a file that
     contradicts *itself* -- ragged row arrays, a conf width that is not top_k, fewer
     candidates than rows, impossible example counts, a baseline carrying the other
     baseline's hyperparameter, a mangled commit -- is refused too, with the handle
     closed on the way out;
  7. `eval.fit_probes` refuses a legacy (schema 1) file before fitting anything;
  8. collection never reuses and never clobbers: an existing output is moved to a
     timestamped backup first.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.
It uses a stub dLLM and temporary files -- no checkpoints, no downloads.

Run with:  python tests/test_actionable_probe.py
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import (  # noqa: E402 - needs the sys.path line above
    generate_unified,
)
from eval.collect_probe_data import (  # noqa: E402
    DATA_KEYS,
    PROBE_ROW_SUPPORT,
    PROBE_SCHEMA_VERSION,
    PROVENANCE_KEYS,
    STEPS_NA,
    THRESHOLD_NA,
    backup_existing,
    build_parser,
    load_probe_npz,
    resolve_out_path,
    select_probe_rows,
    validate_args,
    validate_config,
    validate_probe_npz,
)

MASK_ID = 7
VOCAB = 8


class StubDLLM:
    """Frozen-dLLM stand-in with a constant confidence and a step-dependent argmax.

    Returns log-probabilities as logits, so `generate_unified`'s softmax hands the
    decode loop exactly `confidence`. Confidence is uniform across positions, so
    Fast-dLLM's forced fallback commits exactly one position per step and blocks fill
    one token at a time -- deterministic enough to count.

    The argmax token is a function of how many positions are already committed, so it
    CHANGES from step to step (and is never `MASK_ID`, so a committed position stays
    committed). That is what gives the alignment test teeth: with a constant prediction,
    an x0_history off by one step would still match.
    """

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self.dtype = torch.float32
        self.config = SimpleNamespace(hidden_size=8)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        rest = math.log((1.0 - self.confidence) / (VOCAB - 1))
        logits = torch.full((B, S, VOCAB), rest)
        token = 1 + (input_ids != MASK_ID).sum(dim=-1) % 5  # (B,), in 1..5
        logits.scatter_(
            2, token.view(B, 1, 1).expand(B, S, 1), math.log(self.confidence)
        )
        return SimpleNamespace(logits=logits)


def run_probe(confidence=0.5, remasking="fastdllm", **kwargs):
    """Decode a tiny batch with probe recording on and return the result."""
    prompt = torch.zeros((kwargs.pop("batch", 2), 2), dtype=torch.long)
    return generate_unified(
        model=StubDLLM(confidence),
        prompt=prompt,
        remasking=remasking,
        gen_length=kwargs.pop("gen_length", 8),
        block_length=kwargs.pop("block_length", 4),
        mask_id=MASK_ID,
        record_probe_data=True,
        **kwargs,
    )


# --------------------------------------------------------------------------------
# 1. Time alignment
# --------------------------------------------------------------------------------


class TestTimeAlignment(unittest.TestCase):
    def test_histories_share_one_shape(self):
        r = run_probe(thres=0.9)
        self.assertIsNotNone(r.actionable_history)
        self.assertEqual(r.state_history.shape, r.x0_history.shape)
        self.assertEqual(r.state_history.shape, r.actionable_history.shape)
        self.assertEqual(r.actionable_history.dtype, torch.bool)

    def test_first_recorded_state_is_all_mask(self):
        """Recording happens before the step is applied, so t=0 predates any commit."""
        r = run_probe(thres=0.9, gen_length=8, block_length=4)
        self.assertTrue(bool((r.state_history[:, 0] == MASK_ID).all()))
        # ...and the first step's candidates are exactly the first block.
        self.assertEqual(r.actionable_history[0, 0].tolist(), [True] * 4 + [False] * 4)

    def test_committed_tokens_come_from_the_same_timestep(self):
        """state[t+1] differs from state[t] only at actionable positions, with x0[t].

        This is the alignment claim stated as an equation: if `actionable_history` were
        recorded one step late (after the update) or one step early relative to x0, some
        newly committed position would fall outside the recorded actionable set, or
        carry a token from the wrong step.
        """
        r = run_probe(confidence=0.5, thres=0.9, gen_length=8, block_length=4)
        states, x0s, act = r.state_history, r.x0_history, r.actionable_history
        T = states.shape[1]
        self.assertGreater(T, 1)
        for t in range(T - 1):
            changed = states[:, t + 1] != states[:, t]
            self.assertTrue(bool(changed.any()), f"no progress at step {t}")
            # Every change was an actionable position...
            self.assertTrue(bool((changed & ~act[:, t]).sum() == 0))
            # ...and committed exactly the token x0 predicted at that same step.
            self.assertTrue(bool((states[:, t + 1][changed] == x0s[:, t][changed]).all()))

    def test_final_sequence_matches_the_last_recorded_step(self):
        r = run_probe(confidence=0.5, thres=0.9, gen_length=8, block_length=4)
        final = r.sequences[:, 2:]
        last_state = r.state_history[:, -1]
        changed = final != last_state
        self.assertTrue(bool((changed & ~r.actionable_history[:, -1]).sum() == 0))


# --------------------------------------------------------------------------------
# 2. Current block only
# --------------------------------------------------------------------------------


class TestCurrentBlockOnly(unittest.TestCase):
    def test_actionable_never_leaves_the_current_block(self):
        """Confidence below thres forces one position per step, so blocks fill slowly.

        gen_length 8 / block_length 4 gives 4 steps in block [0,4) then 4 in [4,8).
        The tail of the sequence is masked the whole time and must never be actionable.
        """
        r = run_probe(confidence=0.5, thres=0.9, gen_length=8, block_length=4)
        act = r.actionable_history
        self.assertEqual(act.shape[1], 8)
        for t in range(4):
            self.assertTrue(bool((act[:, t, 4:] == 0).all()), f"step {t} leaked right")
        for t in range(4, 8):
            self.assertTrue(bool((act[:, t, :4] == 0).all()), f"step {t} leaked left")

    def test_actionable_shrinks_as_the_block_fills(self):
        r = run_probe(confidence=0.5, thres=0.9, gen_length=8, block_length=4)
        act = r.actionable_history
        counts = [int(act[0, t].sum()) for t in range(act.shape[1])]
        self.assertEqual(counts, [4, 3, 2, 1, 4, 3, 2, 1])

    def test_a_fully_parallel_block_is_actionable_exactly_once(self):
        """Confidence above thres finishes each block in one step."""
        r = run_probe(confidence=0.99, thres=0.9, gen_length=8, block_length=4)
        act = r.actionable_history
        self.assertEqual(act.shape[1], 2)
        self.assertEqual([int(act[0, t].sum()) for t in range(2)], [4, 4])
        self.assertTrue(bool(act[0, 0, :4].all()))
        self.assertTrue(bool(act[0, 1, 4:].all()))

    def test_masked_positions_outside_the_block_are_excluded(self):
        """The rows schema 1 kept and schema 2 drops: masked, but not yet reachable."""
        r = run_probe(confidence=0.5, thres=0.9, gen_length=8, block_length=4)
        masked = r.state_history == MASK_ID
        self.assertGreater(int((masked & ~r.actionable_history).sum()), 0)


# --------------------------------------------------------------------------------
# 3. Actionable is a subset of the masked state
# --------------------------------------------------------------------------------


class TestActionableIsMasked(unittest.TestCase):
    def _assert_subset(self, r):
        masked = r.state_history == MASK_ID
        self.assertTrue(bool((r.actionable_history & ~masked).sum() == 0))

    def test_subset_under_fastdllm(self):
        self._assert_subset(run_probe(confidence=0.5, thres=0.9))
        self._assert_subset(run_probe(confidence=0.99, thres=0.9))

    def test_subset_under_random(self):
        r = run_probe(remasking="random", steps=8, gen_length=8, block_length=4)
        self.assertIsNotNone(r.actionable_history)
        self._assert_subset(r)

    def test_random_records_the_same_block_local_support(self):
        r = run_probe(remasking="random", steps=8, gen_length=8, block_length=4)
        act = r.actionable_history
        for t in range(4):
            self.assertTrue(bool((act[:, t, 4:] == 0).all()))
        for t in range(4, act.shape[1]):
            self.assertTrue(bool((act[:, t, :4] == 0).all()))


# --------------------------------------------------------------------------------
# 4. Collector row support: actionable AND decided
# --------------------------------------------------------------------------------


class TestSelectProbeRows(unittest.TestCase):
    def test_keeps_only_actionable_and_decided(self):
        state = torch.tensor([[MASK_ID, MASK_ID, MASK_ID, 1]])
        actionable = torch.tensor([[True, True, False, False]])
        x0 = torch.tensor([[1, 2, 1, 1]])
        # position 1 never got decoded: still mask_id at the end
        final = torch.tensor([[1, MASK_ID, 2, 1]])

        sel, lab = select_probe_rows(actionable, state, x0, final, MASK_ID)

        # pos 0: actionable + decided -> kept. pos 1: actionable but undecided -> out.
        # pos 2: decided but not actionable (masked, other block) -> out.
        # pos 3: already committed -> not actionable, out.
        self.assertEqual(sel.tolist(), [[True, False, False, False]])
        self.assertEqual(lab.tolist(), [[True, False, False, False]])

    def test_label_is_zero_when_x0_disagrees_with_the_final_token(self):
        state = torch.tensor([[MASK_ID, MASK_ID]])
        actionable = torch.tensor([[True, True]])
        x0 = torch.tensor([[1, 2]])
        final = torch.tensor([[1, 1]])
        sel, lab = select_probe_rows(actionable, state, x0, final, MASK_ID)
        self.assertEqual(sel.tolist(), [[True, True]])
        self.assertEqual(lab.tolist(), [[True, False]])

    def test_rejects_an_actionable_position_that_is_not_masked(self):
        state = torch.tensor([[1, MASK_ID]])
        actionable = torch.tensor([[True, True]])
        with self.assertRaises(AssertionError):
            select_probe_rows(
                actionable, state, torch.tensor([[1, 1]]), torch.tensor([[1, 1]]),
                MASK_ID,
            )

    def test_end_to_end_selection_is_never_larger_than_the_actionable_set(self):
        r = run_probe(confidence=0.5, thres=0.9, gen_length=8, block_length=4)
        final = r.sequences[:, 2:]
        for t in range(r.state_history.shape[1]):
            sel, lab = select_probe_rows(
                r.actionable_history[:, t],
                r.state_history[:, t],
                r.x0_history[:, t],
                final,
                MASK_ID,
            )
            self.assertLessEqual(int(sel.sum()), int(r.actionable_history[:, t].sum()))
            self.assertTrue(bool((lab & ~sel).sum() == 0))


# --------------------------------------------------------------------------------
# 5. Unsupported modes
# --------------------------------------------------------------------------------


class TestUnsupportedGenerationModes(unittest.TestCase):
    def test_adaptive_block_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            run_probe(thres=0.9, batch=1, adaptive_block=True)
        self.assertIn("adaptive_block", str(cm.exception))

    def test_block_policy_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            run_probe(remasking="block_policy", sampling_mode="categorical")
        self.assertIn("block_policy", str(cm.exception))

    def test_block_schedule_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            run_probe(remasking="block_schedule", block_schedule=((4, 0.9),))
        self.assertIn("block_schedule", str(cm.exception))


class TestCollectorValidation(unittest.TestCase):
    """All of these must fail before an 8B checkpoint is touched."""

    def _args(self, **overrides):
        argv = ["--config", "cfg.yaml", "--out", "out.npz"]
        for k, v in overrides.pop("cli", {}).items():
            argv += [f"--{k}", str(v)]
        return build_parser().parse_args(argv)

    def test_fastdllm_requires_a_threshold(self):
        args = self._args()
        self.assertEqual(args.remasking, "fastdllm")
        self.assertIsNone(args.thres)
        with self.assertRaises(ValueError) as cm:
            validate_args(args)
        self.assertIn("--thres", str(cm.exception))

    def test_random_requires_steps(self):
        args = self._args(cli={"remasking": "random"})
        with self.assertRaises(ValueError) as cm:
            validate_args(args)
        self.assertIn("--steps", str(cm.exception))

    def test_supported_arms_validate(self):
        validate_args(self._args(cli={"remasking": "fastdllm", "thres": 0.9}))
        validate_args(self._args(cli={"remasking": "random", "steps": 256}))

    def test_irrelevant_hyperparameter_is_refused(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(cli={"remasking": "random", "steps": 256,
                                          "thres": 0.9}))

    def test_unsupported_remasking_is_refused_by_the_parser(self):
        for bad in ("policy", "block_policy", "block_schedule", "low_confidence"):
            with self.assertRaises(SystemExit):
                self._args(cli={"remasking": bad})

    def test_unsupported_remasking_is_refused_by_validate_args(self):
        args = self._args(cli={"remasking": "fastdllm", "thres": 0.9})
        args.remasking = "block_policy"
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_adaptive_block_config_is_refused(self):
        cfg = SimpleNamespace(adaptive_block=True, remasking="fastdllm")
        with self.assertRaises(ValueError) as cm:
            validate_config(cfg)
        self.assertIn("adaptive_block", str(cm.exception))

    def test_block_policy_config_is_refused(self):
        for bad in ("block_policy", "block_schedule"):
            with self.assertRaises(ValueError):
                validate_config(SimpleNamespace(adaptive_block=False, remasking=bad))

    def test_nonpositive_steps_is_refused(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError) as cm:
                validate_args(self._args(cli={"remasking": "random", "steps": bad}))
            self.assertIn("--steps", str(cm.exception))

    def test_nonpositive_sizes_are_refused(self):
        for flag in ("n_test", "batch_size", "max_rows"):
            for bad in (0, -8):
                with self.assertRaises(ValueError) as cm:
                    validate_args(
                        self._args(
                            cli={"remasking": "fastdllm", "thres": 0.9, flag: bad}
                        )
                    )
                self.assertIn(f"--{flag}", str(cm.exception))

    def test_a_plain_training_config_passes(self):
        validate_config(SimpleNamespace(adaptive_block=False, remasking="policy"))


# --------------------------------------------------------------------------------
# 6. NPZ schema and provenance
# --------------------------------------------------------------------------------


def make_npz(path, n=8, **overrides):
    """A minimal but complete schema-2 file, with any field overridable."""
    payload = dict(
        h=np.zeros((n, 4), dtype=np.float16),
        conf=np.zeros((n, 16), dtype=np.float16),
        label=np.zeros(n, dtype=np.uint8),
        example_id=np.arange(n, dtype=np.int32),
        step=np.zeros(n, dtype=np.int16),
        pos=np.zeros(n, dtype=np.int16),
        schema_version=np.int32(PROBE_SCHEMA_VERSION),
        row_support=PROBE_ROW_SUPPORT,
        remasking="fastdllm",
        threshold=np.float64(0.9),
        decoding_steps=np.int32(STEPS_NA),
        seed=np.int32(42),
        dataset="gsm8k",
        config="configs/experiment_configs/x.yaml",
        n_examples_requested=np.int32(384),
        n_examples=np.int32(n),
        n_examples_with_rows=np.int32(n),
        batch_size=np.int32(16),
        gen_length=np.int32(256),
        block_length=np.int32(32),
        step_stride=np.int32(4),
        rows_per_step=np.int32(96),
        max_rows=np.int32(400_000),
        top_k=np.int32(16),
        n_candidates=np.int64(1234),
        n_rows=np.int64(n),
        git_commit="0" * 40,
        git_tracked_clean="clean",
    )
    drop = overrides.pop("drop", ())
    payload.update(overrides)
    for key in drop:
        payload.pop(key)
    np.savez(path, **payload)
    return path


class TestNpzContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name="probe.npz"):
        return os.path.join(self.tmp.name, name)

    def test_a_complete_file_validates_and_returns_provenance(self):
        data, prov = load_probe_npz(make_npz(self.path()))
        for key in PROVENANCE_KEYS:
            self.assertIn(key, prov)
        for key in DATA_KEYS:
            self.assertIn(key, data.files)
        self.assertEqual(prov["schema_version"], PROBE_SCHEMA_VERSION)
        self.assertEqual(prov["row_support"], PROBE_ROW_SUPPORT)
        self.assertEqual(prov["remasking"], "fastdllm")
        self.assertEqual(len(prov["git_commit"]), 40)
        self.assertEqual(prov["git_tracked_clean"], "clean")

    def test_random_arm_provenance_uses_the_threshold_sentinel(self):
        _, prov = load_probe_npz(
            make_npz(
                self.path(),
                remasking="random",
                threshold=np.float64(THRESHOLD_NA),
                decoding_steps=np.int32(256),
            )
        )
        self.assertEqual(prov["remasking"], "random")
        self.assertEqual(prov["threshold"], THRESHOLD_NA)
        self.assertEqual(prov["decoding_steps"], 256)

    def test_legacy_file_without_a_version_is_refused(self):
        path = make_npz(self.path(), drop=("schema_version",))
        with self.assertRaises(ValueError) as cm:
            load_probe_npz(path)
        self.assertIn("legacy", str(cm.exception))

    def test_wrong_version_is_refused(self):
        path = make_npz(self.path(), schema_version=np.int32(PROBE_SCHEMA_VERSION + 1))
        with self.assertRaises(ValueError) as cm:
            load_probe_npz(path)
        self.assertIn("schema version", str(cm.exception))

    def test_wrong_row_support_is_refused(self):
        path = make_npz(self.path(), row_support="all_masked_positions")
        with self.assertRaises(ValueError) as cm:
            load_probe_npz(path)
        self.assertIn("row support", str(cm.exception))

    def test_missing_row_support_is_refused(self):
        path = make_npz(self.path(), drop=("row_support",))
        with self.assertRaises(ValueError):
            load_probe_npz(path)

    def test_missing_provenance_key_is_refused(self):
        path = make_npz(self.path(), drop=("git_commit",))
        with self.assertRaises(ValueError) as cm:
            load_probe_npz(path)
        self.assertIn("git_commit", str(cm.exception))

    def test_missing_data_key_is_refused(self):
        path = make_npz(self.path(), drop=("conf",))
        with self.assertRaises(ValueError) as cm:
            load_probe_npz(path)
        self.assertIn("conf", str(cm.exception))

    def test_validate_accepts_an_already_open_npz(self):
        with np.load(make_npz(self.path()), allow_pickle=False) as data:
            self.assertEqual(
                validate_probe_npz(data)["row_support"], PROBE_ROW_SUPPORT
            )

    def test_load_closes_the_handle_when_validation_fails(self):
        """The caller gets an exception, not a file object, so nobody else can close it."""
        path = make_npz(self.path(), row_support="all_masked_positions")
        opened = []
        real_load = np.load

        def spy(*a, **kw):
            data = real_load(*a, **kw)
            opened.append(data)
            return data

        np.load = spy
        try:
            with self.assertRaises(ValueError):
                load_probe_npz(path)
        finally:
            np.load = real_load
        self.assertEqual(len(opened), 1)
        with self.assertRaises(Exception):  # reading a closed zip
            opened[0]["label"]


class TestNpzInternalConsistency(unittest.TestCase):
    """A file can carry every key and still describe a dataset that cannot exist."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name="probe.npz"):
        return os.path.join(self.tmp.name, name)

    def _refused(self, needle, **overrides):
        path = make_npz(self.path(), **overrides)
        with self.assertRaises(ValueError) as cm:
            load_probe_npz(path)
        self.assertIn(needle, str(cm.exception))

    def test_row_arrays_of_different_lengths_are_refused(self):
        self._refused("rows", n=8, label=np.zeros(7, dtype=np.uint8))
        self._refused("rows", n=8, h=np.zeros((5, 4), dtype=np.float16))

    def test_n_rows_disagreeing_with_the_arrays_is_refused(self):
        self._refused("n_rows is 9", n=8, n_rows=np.int64(9))

    def test_empty_file_is_refused(self):
        self._refused("n_rows is 0", n=0, n_rows=np.int64(0))

    def test_rank_one_features_are_refused(self):
        self._refused("rank 1", n=8, h=np.zeros(8, dtype=np.float16))

    def test_conf_width_must_be_top_k(self):
        self._refused("top_k", n=8, conf=np.zeros((8, 8), dtype=np.float16))

    def test_fewer_candidates_than_rows_is_refused(self):
        self._refused("n_candidates", n=8, n_candidates=np.int64(7))

    def test_more_examples_than_requested_is_refused(self):
        self._refused(
            "n_examples 9",
            n=8,
            n_examples=np.int32(9),
            n_examples_requested=np.int32(8),
        )

    def test_more_examples_with_rows_than_processed_is_refused(self):
        self._refused(
            "n_examples_with_rows",
            n=8,
            n_examples=np.int32(4),
            n_examples_with_rows=np.int32(8),
        )

    def test_example_ids_must_match_the_recorded_count(self):
        self._refused(
            "distinct examples",
            n=8,
            example_id=np.zeros(8, dtype=np.int32),  # one example, not eight
        )

    def test_example_ids_outside_the_processed_range_are_refused(self):
        self._refused(
            "outside",
            n=8,
            example_id=np.arange(100, 108, dtype=np.int32),
            n_examples=np.int32(8),
            n_examples_with_rows=np.int32(8),
        )

    def test_a_short_run_that_stopped_early_still_validates(self):
        """--max_rows stopping at 40 of 384 examples is legal, just smaller."""
        _, prov = load_probe_npz(
            make_npz(
                self.path(),
                n=8,
                n_examples=np.int32(40),
                n_examples_with_rows=np.int32(8),
            )
        )
        self.assertEqual(prov["n_examples"], 40)
        self.assertEqual(prov["n_examples_with_rows"], 8)

    def test_unsupported_remasking_is_refused(self):
        self._refused("remasking", remasking="block_policy")

    def test_fastdllm_carrying_decoding_steps_is_refused(self):
        self._refused("decoding_steps", decoding_steps=np.int32(256))

    def test_fastdllm_without_a_threshold_is_refused(self):
        self._refused("threshold", threshold=np.float64(THRESHOLD_NA))

    def test_random_carrying_a_threshold_is_refused(self):
        self._refused(
            "threshold",
            remasking="random",
            threshold=np.float64(0.9),
            decoding_steps=np.int32(256),
        )

    def test_random_without_steps_is_refused(self):
        self._refused(
            "decoding_steps",
            remasking="random",
            threshold=np.float64(THRESHOLD_NA),
            decoding_steps=np.int32(STEPS_NA),
        )

    def test_bad_git_status_is_refused(self):
        self._refused("git_tracked_clean", git_tracked_clean="probably fine")

    def test_short_or_nonhex_commit_is_refused(self):
        self._refused("git_commit", git_commit="0" * 7)
        self._refused("git_commit", git_commit="z" * 40)

    def test_unknown_git_provenance_is_allowed(self):
        _, prov = load_probe_npz(
            make_npz(self.path(), git_commit="unknown", git_tracked_clean="unknown")
        )
        self.assertEqual(prov["git_commit"], "unknown")


# --------------------------------------------------------------------------------
# 8. Fresh collection: never reuse, never clobber
# --------------------------------------------------------------------------------


class TestOutputBackup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_resolve_out_path_matches_what_savez_writes(self):
        self.assertEqual(resolve_out_path("a/b.npz"), "a/b.npz")
        self.assertEqual(resolve_out_path("a/b"), "a/b.npz")

    def test_missing_file_needs_no_backup(self):
        self.assertIsNone(backup_existing(os.path.join(self.tmp.name, "nope.npz")))

    def test_existing_file_is_moved_aside_intact(self):
        path = os.path.join(self.tmp.name, "probe.npz")
        make_npz(path, n=4)
        backup = backup_existing(path)
        self.assertIsNotNone(backup)
        self.assertFalse(os.path.exists(path))
        self.assertTrue(os.path.exists(backup))
        self.assertTrue(backup.endswith(".bak.npz"))
        # the moved file is still the file, not a truncated copy
        data, prov = load_probe_npz(backup)
        self.assertEqual(int(prov["n_rows"]), 4)
        data.close()

    def test_a_second_backup_does_not_overwrite_the_first(self):
        path = os.path.join(self.tmp.name, "probe.npz")
        make_npz(path, n=4)
        first = backup_existing(path)
        make_npz(path, n=5)
        second = backup_existing(path)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first))
        self.assertTrue(os.path.exists(second))


# --------------------------------------------------------------------------------
# 7. The fitter refuses stale data
# --------------------------------------------------------------------------------


class TestFitterRejectsOldSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run_fit(self, path):
        from eval import fit_probes

        argv = sys.argv
        sys.argv = ["fit_probes", "--data", path]
        try:
            fit_probes.main()
        finally:
            sys.argv = argv

    def test_legacy_file_is_refused_before_any_fitting(self):
        path = make_npz(
            os.path.join(self.tmp.name, "legacy.npz"), drop=("schema_version",)
        )
        with self.assertRaises(ValueError) as cm:
            self._run_fit(path)
        self.assertIn("legacy", str(cm.exception))

    def test_wrong_support_is_refused_before_any_fitting(self):
        path = make_npz(
            os.path.join(self.tmp.name, "old.npz"), row_support="all_masked_positions"
        )
        with self.assertRaises(ValueError):
            self._run_fit(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
