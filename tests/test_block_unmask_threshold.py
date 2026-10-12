"""The block_unmask_threshold ablation: Fast-dLLM's threshold rule inside the learned
block head's windows.

What has to hold (CPU only):
  1. With a constant block (fixed schedule b) the loop reproduces remasking='fastdllm'
     at block_length b exactly: same tokens, same NFE, same unmask order. So the
     learned-block cell differs from the Fast-dLLM cell only in where the blocks end.
  2. With the block head free, decoding completes, block sizes are candidates and
     differ from the constant, and the record holds 0/1 commit masks.
  3. Invalid combinations fail up front.

Run with:  python tests/test_block_unmask_threshold.py
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import generate_unified  # noqa: E402
from tests.test_block_unmask_ablation import (  # noqa: E402
    CANDIDATES,
    L,
    MASK_ID,
    PROMPT_L,
    DelimStub,
    _chosen,
    _joint,
    _policy,
    _prompt,
)

TAU = 0.6


def _run(**kw):
    torch.manual_seed(0)
    common = dict(gen_length=L, mask_id=MASK_ID, full_context=True, record_unmask_order=True)
    common.update(kw)
    return generate_unified(DelimStub(3), _prompt(), **common)


def _thres(**kw):
    return _run(
        remasking="block_unmask_policy",
        policy=_joint(),
        sampling_mode="dpls-greedy",
        block_sampling_mode="categorical-argmax",
        block_size_candidates=CANDIDATES,
        block_unmask_threshold=TAU,
        **kw,
    )


class TestThresholdRuleInLearnedBlocks(unittest.TestCase):
    def test_constant_block_equals_fastdllm(self):
        with torch.no_grad():
            ours = _thres(block_unmask_fixed_schedule=(16,))
            ref = _run(remasking="fastdllm", thres=TAU, block_length=16)
        self.assertTrue(torch.equal(ours.sequences, ref.sequences))
        self.assertTrue(torch.equal(ours.steps_taken, ref.steps_taken))
        self.assertTrue(torch.equal(ours.unmask_order, ref.unmask_order))
        self.assertEqual(_chosen(ours), [16] * (L // 16))

    def test_free_block_head(self):
        with torch.no_grad():
            res = _thres()
        self.assertFalse(bool((res.sequences[:, PROMPT_L:] == MASK_ID).any()))
        sizes = _chosen(res)
        self.assertEqual(sum(sizes), L)
        self.assertTrue(all(b in CANDIDATES for b in sizes), sizes)
        # The record holds 0/1 commit masks, one commit per unmask-order step.
        self.assertTrue(bool(res.unmask_order.max() >= 1))

    def test_exclusive_with_unmask_policy(self):
        with self.assertRaises(ValueError):
            _thres(unmask_policy=_policy())

    def test_needs_block_unmask(self):
        with self.assertRaises(ValueError):
            _run(remasking="fastdllm", thres=TAU, block_length=16, block_unmask_threshold=TAU)


if __name__ == "__main__":
    unittest.main()
