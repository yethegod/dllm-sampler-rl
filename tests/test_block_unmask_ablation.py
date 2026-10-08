"""Tests for the ablation swaps of the block_unmask_policy loop.

The ablation grid crosses two unmask heads (the joint policy's own, and the paper's
dit_confidence policy trained at b=32) with three block rules (fixed b, AdaBlock,
the joint policy's learned block head). The loop gets two eval-only switches for it:
``adaptive_block`` (AdaBlock picks each block) and ``unmask_policy`` (a separately
trained dit_confidence policy unmasks). What has to hold, none of which needs a GPU:
  1. With the dit_confidence head swapped in and a constant block, the loop reproduces
     remasking='policy' at that block length exactly: same tokens, same NFE, same
     unmask order. So the learned-block cell differs from the fixed-b cell only in
     where the blocks end.
  2. Same for AdaBlock: the swapped-in head under adaptive_block reproduces
     remasking='policy' with adaptive_block, block sizes included.
  3. The joint head under AdaBlock completes, records the real (non-candidate) block
     lengths, and really uses the joint head (differs from the swapped one).
  4. Invalid combinations fail up front.

Run with:  python tests/test_block_unmask_ablation.py
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import generate_unified  # noqa: E402
from common.models.policy import DiTBlockUnmaskPolicy, PolicyHFWrapper  # noqa: E402
from tests.test_policy_dpls_greedy import (  # noqa: E402
    DELIM,
    L,
    MASK_ID,
    PROMPT_L,
    StubDLLM,
    _policy,
)

CANDIDATES = (8, 16, 32)
ADA = dict(adaptive_block=True, delimiter_ids=(DELIM,), delimiter_threshold=0.05)


class DelimStub(StubDLLM):
    """StubDLLM whose argmax is the delimiter at ~1 in 9 positions per forward.

    The plain stub's random logits almost never predict DELIM confidently, so AdaBlock
    would always fall back to B0 and the tests would not exercise the delimiter rule.
    """

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        g = torch.Generator().manual_seed(7 * 100003 + self.calls)
        out = super().__call__(input_ids, attention_mask, output_hidden_states)
        hit = torch.rand(input_ids.shape[1], generator=g) < 0.11
        out.logits[:, hit, DELIM] = 30.0
        return out


def _joint(seed=1):
    torch.manual_seed(seed)
    core = DiTBlockUnmaskPolicy(
        block_size_candidates=CANDIDATES,
        window_cond=True,
        boundary_init_gain=1.0,
        hidden_dim=16,
        feedforward_dim=32,
        num_heads=2,
        time_embed_dim=16,
        confidences_top_p=1,
        smart_init=-2.0,
    )
    with torch.no_grad():
        core.window_embedding.weight.normal_(0, 0.5)
    return PolicyHFWrapper(core, "dit_block_unmask").eval()


def _prompt():
    return torch.randint(0, MASK_ID, (1, PROMPT_L), generator=torch.Generator().manual_seed(1))


def _run(rng_seed=0, **kw):
    torch.manual_seed(rng_seed)
    common = dict(
        gen_length=L,
        mask_id=MASK_ID,
        full_context=True,
        record_unmask_order=True,
    )
    common.update(kw)
    return generate_unified(DelimStub(3), _prompt(), **common)


def _swapped(mode, **kw):
    return _run(
        remasking="block_unmask_policy",
        policy=_joint(),
        unmask_policy=_policy(),
        sampling_mode=mode,
        block_sampling_mode="categorical-argmax",
        block_size_candidates=CANDIDATES,
        **kw,
    )


def _reference(mode, **kw):
    return _run(remasking="policy", policy=_policy(), sampling_mode=mode, **kw)


def _chosen(res):
    return res.block_sizes_chosen[0][res.block_decisions[0]].tolist()


class TestSwappedHeadMatchesPolicyLoop(unittest.TestCase):
    def _assert_same(self, a, b, tag):
        self.assertTrue(torch.equal(a.sequences, b.sequences), tag)
        self.assertTrue(torch.equal(a.steps_taken, b.steps_taken), tag)
        self.assertTrue(torch.equal(a.unmask_order, b.unmask_order), tag)

    def test_constant_block(self):
        for mode in ("dpls-greedy", "dpls"):
            with torch.no_grad():
                swapped = _swapped(mode, block_unmask_fixed_schedule=(16,))
                ref = _reference(mode, block_length=16)
            self._assert_same(swapped, ref, mode)
            self.assertEqual(_chosen(swapped), [16] * (L // 16))

    def test_adaptive_block(self):
        with torch.no_grad():
            swapped = _swapped("dpls-greedy", block_length=16, **ADA)
            ref = _reference("dpls-greedy", block_length=16, **ADA)
        self._assert_same(swapped, ref, "ada")
        self.assertEqual(_chosen(swapped), ref.block_sizes)
        # The stub must actually exercise the delimiter rule, not only the fallback.
        self.assertTrue(any(b != 16 for b in ref.block_sizes), ref.block_sizes)


class TestJointHeadAdaptive(unittest.TestCase):
    def test_completes_with_real_lengths(self):
        with torch.no_grad():
            res = _run(
                remasking="block_unmask_policy",
                policy=_joint(),
                sampling_mode="dpls-greedy",
                block_sampling_mode="categorical-argmax",
                block_size_candidates=CANDIDATES,
                block_length=16,
                **ADA,
            )
        self.assertFalse(bool((res.sequences[:, PROMPT_L:] == MASK_ID).any()))
        sizes = _chosen(res)
        self.assertEqual(sum(sizes), L)
        self.assertTrue(any(b not in CANDIDATES for b in sizes), sizes)
        # Blocks fill in order: nothing of block k is committed after block k+1 starts.
        order = res.unmask_order[0]
        edges = [0]
        for b in sizes:
            edges.append(edges[-1] + b)
        for lo, mid, hi in zip(edges, edges[1:], edges[2:]):
            self.assertLess(int(order[lo:mid].max()), int(order[mid:hi].min()))

    def test_uses_its_own_head(self):
        with torch.no_grad():
            own = _run(
                remasking="block_unmask_policy",
                policy=_joint(),
                sampling_mode="dpls-greedy",
                block_sampling_mode="categorical-argmax",
                block_size_candidates=CANDIDATES,
                block_length=16,
                **ADA,
            )
            swapped = _swapped("dpls-greedy", block_length=16, **ADA)
        self.assertFalse(torch.equal(own.unmask_order, swapped.unmask_order))


class TestValidation(unittest.TestCase):
    def test_adaptive_and_fixed_schedule_conflict(self):
        with self.assertRaises(ValueError):
            _swapped("dpls-greedy", block_unmask_fixed_schedule=(16,), **ADA)

    def test_unmask_policy_needs_block_unmask(self):
        with self.assertRaises(ValueError):
            _reference("dpls-greedy", block_length=16, unmask_policy=_policy())

    def test_unmask_policy_must_be_confidence_policy(self):
        with self.assertRaises(ValueError):
            _run(
                remasking="block_unmask_policy",
                policy=_joint(),
                unmask_policy=_joint(),
                sampling_mode="dpls-greedy",
                block_sampling_mode="categorical-argmax",
                block_size_candidates=CANDIDATES,
            )


if __name__ == "__main__":
    unittest.main()
