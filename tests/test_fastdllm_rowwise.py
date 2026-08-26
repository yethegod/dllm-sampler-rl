"""Regression tests for the row-wise Fast-dLLM fallback in fixed-block decoding.

Fast-dLLM unmasks every masked position in the current block whose confidence clears
`thres`, and forces one position when nothing clears -- otherwise a block would never
complete. The forcing rule used to be batch-wide (`if not unmask_local.any()`), which is
only correct at batch size 1: with B > 1, one row clearing the threshold suppressed the
forced token for *every* stalled row, so a stalled row made no progress that step. The
fixed-block path now shares `_confidence_threshold_unmask_rowwise` with the block_policy
path, which forces per row.

What is pinned here:

  1. a stalled row still gets exactly one forced position when a sibling row progresses;
  2. a row with no eligible candidates (finished, or block already full) gets no action,
     so a committed token is never overwritten;
  3. only positions inside the current block are ever eligible, above threshold or
     forced;
  4. every active row progresses -- checked on random inputs and end to end through
     `generate_unified` against a stub dLLM, where the old rule leaves masked tokens
     behind;
  5. at B = 1 the new rule reproduces the old fixed-block logic exactly, so previously
     collected baselines (all batch size 1) stay reproducible;
  6. thresholds accept scalars, 0-D tensors and (B, 1), and reject (B,) -- which torch
     would broadcast over positions instead of rows -- with a clear ValueError.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.

Run with:  python tests/test_fastdllm_rowwise.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import (  # noqa: E402 - needs the sys.path line above
    _confidence_threshold_unmask,
    _confidence_threshold_unmask_rowwise,
    _normalize_row_threshold,
    generate_unified,
)


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------


def probs_from_confidence(conf: torch.Tensor) -> torch.Tensor:
    """(B, L) confidences in [0.5, 1] -> a (B, L, 2) distribution with that max prob."""
    return torch.stack([1.0 - conf, conf], dim=-1)


def legacy_fixed_block_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    block_slice: slice,
    thres: float,
) -> torch.Tensor:
    """The pre-change fixed-block rule, verbatim, as the B = 1 reference."""
    confidence = probs.max(dim=-1).values

    confidence_masked = confidence[:, block_slice].clone()
    confidence_masked[~block_mask_index] = -torch.inf

    unmask_local = confidence_masked > thres
    if not unmask_local.any():
        force_idx = torch.argmax(confidence_masked, dim=-1)
        unmask_local.scatter_(1, force_idx.unsqueeze(-1), True)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask


class StubDLLM:
    """Frozen-dLLM stand-in whose confidence is a constant per row.

    Returns log-probabilities as logits, so `generate_unified`'s softmax hands the
    decode loop exactly the confidences configured here. The argmax token is always
    index 1, which is never `mask_id`, so every unmasked position stays unmasked.
    """

    def __init__(self, row_confidence: torch.Tensor):
        self.row_confidence = row_confidence
        self.dtype = torch.float32
        self.config = SimpleNamespace(hidden_size=8)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        conf = self.row_confidence.view(B, 1).expand(B, S)
        return SimpleNamespace(logits=probs_from_confidence(conf).log())


# --------------------------------------------------------------------------------
# Fallback behaviour: stalled vs progressing rows
# --------------------------------------------------------------------------------


class TestRowWiseFallback(unittest.TestCase):
    def test_stalled_row_progresses_while_sibling_clears_threshold(self):
        """Row 0 clears the threshold; row 1 must still get its one forced token."""
        block = slice(0, 4)
        conf = torch.tensor(
            [
                [0.95, 0.99, 0.60, 0.55, 0.99],  # row 0: two positions clear 0.9
                [0.60, 0.55, 0.80, 0.70, 0.99],  # row 1: nothing clears
            ]
        )
        block_mask_index = torch.ones((2, 4), dtype=torch.bool)

        unmask = _confidence_threshold_unmask(
            block_mask_index, probs_from_confidence(conf), block, 0.9
        )

        # Row 0: exactly the above-threshold in-block positions.
        self.assertEqual(unmask[0].tolist(), [True, True, False, False, False])
        # Row 1: exactly one forced action, at its most confident in-block position.
        self.assertEqual(int(unmask[1].sum()), 1)
        self.assertTrue(bool(unmask[1, 2]))

    def test_every_stalled_row_gets_exactly_one_action(self):
        """Several stalled rows at once: one action each, none starving another."""
        block = slice(2, 6)
        conf = torch.full((4, 8), 0.6)
        conf[0, 3] = 0.99  # the only row that clears
        conf[1, 4] = 0.75  # stalled, argmax inside the block
        conf[2, 2] = 0.70
        conf[3, 5] = 0.65
        block_mask_index = torch.ones((4, 4), dtype=torch.bool)

        unmask = _confidence_threshold_unmask(
            block_mask_index, probs_from_confidence(conf), block, 0.9
        )

        self.assertEqual(unmask[0].nonzero().flatten().tolist(), [3])
        self.assertEqual(unmask[1].nonzero().flatten().tolist(), [4])
        self.assertEqual(unmask[2].nonzero().flatten().tolist(), [2])
        self.assertEqual(unmask[3].nonzero().flatten().tolist(), [5])

    def test_forced_position_is_the_most_confident_candidate(self):
        """The forced token goes to the argmax among *eligible* positions only."""
        block = slice(0, 4)
        conf = torch.tensor([[0.60, 0.88, 0.70, 0.55]])
        block_mask_index = torch.tensor([[True, False, True, True]])  # pos 1 committed

        unmask = _confidence_threshold_unmask(
            block_mask_index, probs_from_confidence(conf), block, 0.9
        )

        # Not position 1 (highest confidence but already unmasked): position 2.
        self.assertEqual(unmask[0].nonzero().flatten().tolist(), [2])


# --------------------------------------------------------------------------------
# Finished rows and rows with no eligible candidates
# --------------------------------------------------------------------------------


class TestFinishedRows(unittest.TestCase):
    def test_row_with_no_candidates_gets_no_action(self):
        """A row whose block is already full is never forced, so nothing is clobbered."""
        block = slice(0, 4)
        conf = torch.tensor([[0.60, 0.60, 0.60, 0.60], [0.99, 0.99, 0.99, 0.99]])
        block_mask_index = torch.tensor(
            [
                [True, True, True, True],  # active, stalled
                [False, False, False, False],  # block already complete
            ]
        )

        unmask = _confidence_threshold_unmask(
            block_mask_index, probs_from_confidence(conf), block, 0.9
        )

        self.assertEqual(int(unmask[0].sum()), 1)
        self.assertFalse(bool(unmask[1].any()))

    def test_whole_batch_finished_produces_no_action(self):
        """Where the old rule forced position 0 for every row, there is now no action.

        This is the one deliberate divergence from the old fixed-block logic. It is
        unreachable from the decode loop, which breaks out before calling this on an
        exhausted block -- and forcing there would have overwritten a committed token.
        """
        block = slice(0, 4)
        conf = torch.full((2, 4), 0.99)
        block_mask_index = torch.zeros((2, 4), dtype=torch.bool)
        probs = probs_from_confidence(conf)

        unmask = _confidence_threshold_unmask(block_mask_index, probs, block, 0.9)
        self.assertFalse(bool(unmask.any()))

        legacy = legacy_fixed_block_unmask(block_mask_index, probs, block, 0.9)
        self.assertTrue(bool(legacy[:, 0].all()))  # the old rule did clobber position 0


# --------------------------------------------------------------------------------
# Block boundaries
# --------------------------------------------------------------------------------


class TestBlockBoundaries(unittest.TestCase):
    def test_out_of_block_positions_never_unmasked(self):
        """Confident masked positions outside the block stay masked, in both paths."""
        block = slice(4, 8)
        conf = torch.full((2, 12), 0.99)  # everything, everywhere, clears
        conf[1, 4:8] = 0.60  # row 1 stalls inside its block
        block_mask_index = torch.ones((2, 4), dtype=torch.bool)

        unmask = _confidence_threshold_unmask(
            block_mask_index, probs_from_confidence(conf), block, 0.9
        )

        self.assertFalse(bool(unmask[:, :4].any()))
        self.assertFalse(bool(unmask[:, 8:].any()))
        self.assertEqual(unmask[0, 4:8].tolist(), [True] * 4)  # threshold path
        self.assertEqual(int(unmask[1, 4:8].sum()), 1)  # forced path

    def test_rows_may_sit_in_different_blocks(self):
        """The row-wise entry point takes a (B, L) block mask, one block per row."""
        conf = torch.full((2, 8), 0.60)
        conf[0, 1] = 0.95
        conf[1, 6] = 0.95
        block_mask_index = torch.zeros((2, 8), dtype=torch.bool)
        block_mask_index[0, 0:4] = True
        block_mask_index[1, 4:8] = True

        unmask = _confidence_threshold_unmask_rowwise(
            block_mask_index, probs_from_confidence(conf), torch.tensor([[0.9], [0.9]])
        )

        self.assertEqual(unmask[0].nonzero().flatten().tolist(), [1])
        self.assertEqual(unmask[1].nonzero().flatten().tolist(), [6])


# --------------------------------------------------------------------------------
# Progress invariants
# --------------------------------------------------------------------------------


class TestProgress(unittest.TestCase):
    def test_every_active_row_progresses_on_random_inputs(self):
        """For random blocks/confidences: in-block only, and no active row stalls."""
        torch.manual_seed(0)
        B, L = 6, 16
        block = slice(4, 12)

        for thres in (0.0, 0.55, 0.9, 1.0):
            for _ in range(25):
                conf = 0.5 + 0.5 * torch.rand(B, L)
                block_mask_index = torch.rand(B, 8) < 0.5
                block_mask_index[0] = False  # always exercise a finished row

                unmask = _confidence_threshold_unmask(
                    block_mask_index, probs_from_confidence(conf), block, thres
                )

                lifted = torch.zeros((B, L), dtype=torch.bool)
                lifted[:, block] = block_mask_index

                # Only ever eligible positions.
                self.assertTrue(bool((unmask <= lifted).all()))
                # Active rows progress; inactive rows do nothing.
                self.assertEqual(
                    unmask.any(dim=-1).tolist(), lifted.any(dim=-1).tolist()
                )

    def test_generate_unified_drains_every_row(self):
        """End to end: a permanently stalled row still finishes alongside a fast one.

        Row 0 clears the threshold every step and takes a whole block at a time; row 1
        never clears and lives entirely on the fallback. Under the old batch-wide rule
        row 1 got nothing on the step row 0 cleared, so it ran out of in-block steps and
        finished with masked tokens left over.
        """
        mask_id = 99
        gen_length, block_length = 8, 4
        prompt = torch.zeros((2, 3), dtype=torch.long)
        model = StubDLLM(torch.tensor([0.99, 0.60]))

        result = generate_unified(
            model=model,
            prompt=prompt,
            remasking="fastdllm",
            thres=0.9,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            mask_id=mask_id,
        )

        completion = result.sequences[:, prompt.shape[1] :]
        self.assertFalse(bool((completion == mask_id).any()))
        # Row 0: one step per block. Row 1: one position per step.
        self.assertEqual(
            result.steps_taken.tolist(), [gen_length // block_length, gen_length]
        )

    def test_generate_unified_per_row_thresholds(self):
        """A (B, 1) threshold applies per row end to end (the expert-steering path)."""
        mask_id = 99
        gen_length, block_length = 8, 4
        prompt = torch.zeros((2, 3), dtype=torch.long)
        model = StubDLLM(torch.tensor([0.8, 0.8]))

        result = generate_unified(
            model=model,
            prompt=prompt,
            remasking="fastdllm",
            thres=torch.tensor([[0.7], [0.9]]),  # row 0 clears, row 1 never does
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            mask_id=mask_id,
        )

        completion = result.sequences[:, prompt.shape[1] :]
        self.assertFalse(bool((completion == mask_id).any()))
        self.assertEqual(
            result.steps_taken.tolist(), [gen_length // block_length, gen_length]
        )


# --------------------------------------------------------------------------------
# B = 1 equivalence with the old fixed-block logic
# --------------------------------------------------------------------------------


class TestBatchOneEquivalence(unittest.TestCase):
    def test_matches_legacy_on_random_single_row_inputs(self):
        torch.manual_seed(1)
        L, BL = 16, 8
        block = slice(4, 12)

        for thres in (0.0, 0.51, 0.6, 0.75, 0.9, 0.999):
            for _ in range(100):
                conf = 0.5 + 0.5 * torch.rand(1, L)
                block_mask_index = torch.rand(1, BL) < 0.5
                if not block_mask_index.any():
                    # The exhausted block is the documented divergence and is
                    # unreachable from the decode loop; covered separately above.
                    continue
                probs = probs_from_confidence(conf)

                new = _confidence_threshold_unmask(
                    block_mask_index, probs, block, thres
                )
                old = legacy_fixed_block_unmask(
                    block_mask_index, probs, block, thres
                )
                self.assertTrue(
                    bool(torch.equal(new, old)),
                    f"thres={thres} conf={conf} mask={block_mask_index}",
                )

    def test_matches_legacy_on_a_full_block(self):
        block = slice(0, 4)
        conf = torch.tensor([[0.95, 0.60, 0.99, 0.55]])
        probs = probs_from_confidence(conf)
        block_mask_index = torch.ones((1, 4), dtype=torch.bool)

        self.assertTrue(
            torch.equal(
                _confidence_threshold_unmask(block_mask_index, probs, block, 0.9),
                legacy_fixed_block_unmask(block_mask_index, probs, block, 0.9),
            )
        )


# --------------------------------------------------------------------------------
# Threshold shapes
# --------------------------------------------------------------------------------


class TestThresholdShapes(unittest.TestCase):
    def setUp(self):
        self.block = slice(0, 4)
        self.conf = torch.tensor([[0.60, 0.95, 0.55, 0.70], [0.60, 0.95, 0.55, 0.70]])
        self.probs = probs_from_confidence(self.conf)
        self.block_mask_index = torch.ones((2, 4), dtype=torch.bool)

    def _unmask(self, thres):
        return _confidence_threshold_unmask(
            self.block_mask_index, self.probs, self.block, thres
        )

    def test_accepts_python_scalar(self):
        self.assertEqual(self._unmask(0.9)[0].nonzero().flatten().tolist(), [1])

    def test_accepts_int_scalar(self):
        # Everything is below an integer threshold of 1: every row falls back.
        unmask = self._unmask(1)
        self.assertEqual(unmask.sum(dim=-1).tolist(), [1, 1])

    def test_accepts_zero_dim_tensor(self):
        self.assertTrue(
            torch.equal(self._unmask(torch.tensor(0.9)), self._unmask(0.9))
        )

    def test_accepts_b_by_one_tensor(self):
        # Identical rows, different thresholds: the looser one clears two positions,
        # the tighter one only the single most confident.
        unmask = self._unmask(torch.tensor([[0.65], [0.9]]))
        self.assertEqual(unmask[0].nonzero().flatten().tolist(), [1, 3])
        self.assertEqual(unmask[1].nonzero().flatten().tolist(), [1])

    def test_accepts_one_by_one_tensor(self):
        self.assertTrue(
            torch.equal(self._unmask(torch.tensor([[0.9]])), self._unmask(0.9))
        )

    def test_rejects_shape_b(self):
        with self.assertRaises(ValueError) as cm:
            self._unmask(torch.tensor([0.5, 0.9]))
        self.assertIn("(2,)", str(cm.exception))

    def test_rejects_shape_one_when_batch_is_one(self):
        # (B,) with B = 1: same ambiguity, same rejection, not silently a scalar.
        with self.assertRaises(ValueError):
            _confidence_threshold_unmask(
                self.block_mask_index[:1],
                self.probs[:1],
                self.block,
                torch.tensor([0.9]),
            )

    def test_rejects_other_ambiguous_shapes(self):
        for bad in (
            torch.full((1, 2), 0.9),  # (1, B): broadcasts over rows
            torch.full((2, 4), 0.9),  # (B, L): per-position, not per-row
            torch.full((3, 1), 0.9),  # wrong batch size
            torch.full((2, 1, 1), 0.9),  # 3-D
        ):
            with self.subTest(shape=tuple(bad.shape)):
                with self.assertRaises(ValueError):
                    self._unmask(bad)

    def test_rowwise_entry_point_validates_too(self):
        with self.assertRaises(ValueError):
            _confidence_threshold_unmask_rowwise(
                self.block_mask_index, self.probs, torch.tensor([0.5, 0.9])
            )

    def test_normalizer_returns_broadcastable_shape(self):
        for thres, expected in (
            (0.9, (1, 1)),
            (torch.tensor(0.9), (1, 1)),
            (torch.tensor([[0.9]]), (1, 1)),
            (torch.tensor([[0.5], [0.9]]), (2, 1)),
        ):
            with self.subTest(thres=thres):
                out = _normalize_row_threshold(thres, 2, torch.device("cpu"))
                self.assertEqual(tuple(out.shape), expected)

    def test_normalizer_moves_accepted_tensors_to_requested_device(self):
        """An accepted tensor threshold lands on `device`, keeping dtype and shape.

        The real failure is a CPU threshold against CUDA probabilities, which needs no
        CUDA to pin: "meta" is likewise a device other than the tensor's own, so a
        normalizer that returned the threshold untouched fails here too.
        """
        for thres, expected in (
            (torch.tensor(0.9), (1, 1)),
            (torch.tensor([[0.9]]), (1, 1)),
            (torch.tensor([[0.5], [0.9]]), (2, 1)),
            (torch.tensor([[0.5], [0.9]], dtype=torch.float64), (2, 1)),
        ):
            with self.subTest(thres=thres):
                out = _normalize_row_threshold(thres, 2, torch.device("meta"))
                self.assertEqual(out.device.type, "meta")
                self.assertEqual(out.dtype, thres.dtype)
                self.assertEqual(tuple(out.shape), expected)


if __name__ == "__main__":
    unittest.main()
