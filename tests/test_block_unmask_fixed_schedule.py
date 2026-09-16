"""Tests for the eval-only probes of the block_unmask_policy loop.

``fixed_schedule`` must replace the block head's draw and nothing else; ``cond_block``
must change only what the unmask head is told (and therefore, for a policy whose
window embedding is non-zero, its logits), never the real block that gets sampled.
Both default off, so the recorded training numerics are untouched.

Run with:  python tests/test_block_unmask_fixed_schedule.py
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import (  # noqa: E402
    _block_unmask_policy_loop,
)
from tests.test_block_unmask_window_cond import (  # noqa: E402
    CANDIDATES,
    L,
    MASK_ID,
    P,
    _make_policy,
    _perturb,
)


def _run(policy, B, fixed_schedule=None, cond_block=None, seed=3):
    torch.manual_seed(seed)
    prompt_L = 3
    V = 5
    x = torch.full((B, prompt_L + L), MASK_ID, dtype=torch.long)
    x[:, :prompt_L] = 1
    steps_taken = torch.zeros(B, dtype=torch.int)
    g = torch.Generator().manual_seed(seed)

    def forward_logits():
        logits = torch.randn((B, L, V), generator=g)
        logits[..., MASK_ID] = -1e9
        probs = torch.softmax(logits, dim=-1)
        return None, probs, probs.argmax(dim=-1)

    rec = _block_unmask_policy_loop(
        x,
        prompt_L,
        L,
        MASK_ID,
        L,
        steps_taken,
        policy,
        forward_logits,
        None,
        CANDIDATES,
        "bernoulli-argmax",  # always finishes, so every block decision is reached
        "categorical",
        P,
        1.0,
        fixed_schedule=fixed_schedule,
        cond_block=cond_block,
    )
    return x, rec


def _chosen(rec, row):
    return rec["block_sizes_chosen"][row][rec["block_decisions"][row]].tolist()


class TestFixedSchedule(unittest.TestCase):
    def setUp(self):
        self.policy = _make_policy(window_cond=True)
        _perturb(self.policy)

    def test_constant_block_is_honoured(self):
        with torch.no_grad():
            _, rec = _run(self.policy, B=2, fixed_schedule=(8,))
        for row in range(2):
            self.assertEqual(_chosen(rec, row), [8] * (L // 8))

    def test_schedule_repeats_last_entry_and_clips_to_fit(self):
        # 16 then 32: the second decision starts at 16 with 16 left, so 32 drops to
        # the largest candidate that fits (16); the sequence is then complete.
        with torch.no_grad():
            _, rec = _run(self.policy, B=1, fixed_schedule=(16, 32))
        self.assertEqual(_chosen(rec, 0), [16, 16])

    def test_block_logits_still_recorded(self):
        with torch.no_grad():
            _, rec = _run(self.policy, B=1, fixed_schedule=(32,))
        dec = rec["block_decisions"][0]
        block_logits = rec["sampling_inputs"][0][dec][:, L:]
        self.assertTrue(torch.isfinite(block_logits).any())
        self.assertFalse((block_logits == 0).all())

    def test_off_by_default(self):
        # No override: whatever gets drawn is feasible and tiles the sequence.
        with torch.no_grad():
            _, rec = _run(self.policy, B=2)
        for row in range(2):
            self.assertEqual(sum(_chosen(rec, row)), L)


class TestCondBlock(unittest.TestCase):
    def setUp(self):
        self.policy = _make_policy(window_cond=True)
        _perturb(self.policy)

    def test_real_block_unchanged_but_head_sees_the_lie(self):
        with torch.no_grad():
            _, plain = _run(self.policy, B=1, fixed_schedule=(32,))
            _, lied = _run(self.policy, B=1, fixed_schedule=(32,), cond_block=8)
        self.assertEqual(_chosen(plain, 0), [32])
        self.assertEqual(_chosen(lied, 0), [32])
        # Recorded block_end is what the head saw: start + 8, not 32.
        end_in = lied["policy_inputs"][4][0, 0, 0].item()
        self.assertEqual(end_in, 8)
        # Sampling mask (real block) still spans all 32 positions at step 0.
        self.assertTrue(lied["sampling_masks"][0, 0, :L].all())
        # And the perturbed window embedding makes the unmask logits differ.
        self.assertFalse(
            torch.allclose(
                plain["sampling_inputs"][0, 0, :L], lied["sampling_inputs"][0, 0, :L]
            )
        )

    def test_zero_window_embedding_is_insensitive(self):
        policy = _make_policy(window_cond=True)  # window_embedding still zero
        with torch.no_grad():
            _, a = _run(policy, B=1, fixed_schedule=(32,))
            _, b = _run(policy, B=1, fixed_schedule=(32,), cond_block=8)
        self.assertTrue(
            torch.allclose(a["sampling_inputs"][0, 0, :L], b["sampling_inputs"][0, 0, :L])
        )

    def test_cond_block_clips_to_sequence_end(self):
        with torch.no_grad():
            _, rec = _run(self.policy, B=1, fixed_schedule=(8,), cond_block=128)
        ends = rec["policy_inputs"][4][0, :, 0]
        self.assertTrue((ends <= L).all())
        self.assertEqual(ends[0].item(), L)


if __name__ == "__main__":
    unittest.main()
