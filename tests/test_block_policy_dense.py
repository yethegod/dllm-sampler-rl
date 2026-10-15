"""Block-size-only policy with a dense (token-level) candidate set and hidden input.

llada_8b_instruct_dit_blocksize_only_hidden_proj_dense.yaml sets
block_size_candidates to every b in 1..128. Two things change against the old
{8,...,128} set:

  1. T_cap = L / min(candidates) becomes L, so remasking='block_policy' can no longer
     preallocate its decision buffers at T_cap (the hidden-state stream alone would be
     (B, 256, L, 4096)). They now start small and double on demand. These tests drive
     a rollout past the initial allocation and check the record is still exact.
  2. block_start no longer stays on a multiple of 8, so the feasibility mask
     (b <= remaining) is what keeps every block inside [0, L).

CPU only; the dLLM is a stub that returns hidden states. Run with
`python tests/test_block_policy_dense.py`.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import generate_unified  # noqa: E402
from common.models.policy import (  # noqa: E402
    DiTBlockSizeHiddenProjPolicy,
    PolicyHFWrapper,
)

MASK_ID = 99
HIDDEN = 16


class StubDLLM:
    """Frozen-dLLM stand-in: constant confidence, argmax token 1, random hidden states."""

    def __init__(self, confidence: float):
        self.confidence = confidence
        self.dtype = torch.float32
        self.config = SimpleNamespace(hidden_size=HIDDEN)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        conf = torch.full((B, S), self.confidence)
        logits = torch.stack([1.0 - conf, conf], dim=-1).log()
        hidden = (torch.randn(B, S, HIDDEN),) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden)

    __call__ = forward


class FixedBlockPolicy(torch.nn.Module):
    """Always picks block size `b`, or the largest size that fits when b does not."""

    policy_type = "dit_block_size_hidden_proj"

    def __init__(self, candidates, b):
        super().__init__()
        self.candidates = torch.tensor(candidates)
        self.b = b

    def forward(self, m, c, timestep, block_start):
        B, L = m.shape
        assert c.shape == (B, L, HIDDEN), c.shape
        # Sharply peaked on b, falling off with distance, so the feasibility mask
        # turns an oversized b into the nearest size that still fits.
        dist = (self.candidates - self.b).abs().float()
        logits = (-50.0 * dist).expand(B, -1).clone()
        end = block_start + self.candidates
        logits = logits.masked_fill(end > L, float("-inf"))
        return logits, torch.zeros(B, 1)


def run(policy, candidates, B=2, L=40, confidence=0.99, sampling_mode="categorical"):
    return generate_unified(
        model=StubDLLM(confidence),
        prompt=torch.zeros((B, 3), dtype=torch.long),
        remasking="block_policy",
        policy=policy,
        gen_length=L,
        temperature=0.0,
        mask_id=MASK_ID,
        sampling_mode=sampling_mode,
        block_size_candidates=tuple(candidates),
        threshold_candidates=(0.9,),
    )


class TestBufferGrowth(unittest.TestCase):
    def test_one_token_blocks_record_every_decision(self):
        """b=1 everywhere: L decisions per row, past the initial 32-slot allocation."""
        L = 40
        cands = list(range(1, 129))
        result = run(FixedBlockPolicy(cands, 1), cands, L=L)

        completion = result.sequences[:, 3:]
        self.assertFalse(bool((completion == MASK_ID).any()))
        self.assertEqual(result.block_decisions.shape, (2, L))
        self.assertTrue(bool(result.block_decisions.all()))
        self.assertTrue(bool((result.block_sizes_chosen == 1).all()))
        # Decision t starts at position t: nothing was lost or overwritten on growth.
        starts = result.policy_inputs[3][..., 0]
        self.assertEqual(starts.tolist(), [list(range(L))] * 2)
        # The hidden stream was recorded at every slot, including the grown ones.
        c = result.policy_inputs[1]
        self.assertEqual(c.shape, (2, L, L, HIDDEN))
        self.assertTrue(bool((c.abs().sum(dim=(-1, -2)) > 0).all()))

    def test_no_growth_matches_preallocated_shape(self):
        """A few large blocks never trigger growth and pack to T = number of blocks."""
        L = 40
        cands = list(range(1, 129))
        result = run(FixedBlockPolicy(cands, 16), cands, L=L)
        # 16 + 16 + 8 (the last one clamped by feasibility).
        self.assertEqual(result.block_sizes_chosen.tolist(), [[16, 16, 8]] * 2)
        self.assertTrue(bool(result.block_decisions.all()))


class TestDensePolicy(unittest.TestCase):
    def make_policy(self, cands):
        torch.manual_seed(0)
        core = DiTBlockSizeHiddenProjPolicy(
            dllm=SimpleNamespace(config=SimpleNamespace(hidden_size=HIDDEN)),
            block_size_candidates=tuple(cands),
            thresholds=(0.9,),
            hidden_dim=32,
            feedforward_dim=64,
            num_heads=2,
            time_embed_dim=32,
            smart_init=-2.0,
        )
        return PolicyHFWrapper(core, "dit_block_size_hidden_proj")

    def test_blocks_tile_exactly_with_random_sizes(self):
        """Sampled token-level sizes always tile [0, L): no overrun, no gap."""
        L = 50
        cands = list(range(1, 129))
        policy = self.make_policy(cands)
        for seed in range(5):
            torch.manual_seed(seed)
            result = run(policy, cands, B=4, L=L)
            self.assertFalse(bool((result.sequences[:, 3:] == MASK_ID).any()))
            sizes = result.block_sizes_chosen * result.block_decisions
            self.assertEqual(sizes.sum(dim=-1).tolist(), [L] * 4)
            starts = result.policy_inputs[3][..., 0]
            for r in range(4):
                n = int(result.block_decisions[r].sum())
                expect = torch.cumsum(sizes[r, :n], 0) - sizes[r, :n]
                self.assertEqual(starts[r, :n].tolist(), expect.tolist())

    def test_infeasible_sizes_get_no_mass(self):
        L = 50
        cands = list(range(1, 129))
        policy = self.make_policy(cands)
        m = torch.ones(3, L, dtype=torch.bool)
        h = torch.randn(3, L, HIDDEN)
        t = torch.zeros(3, 1)
        start = torch.tensor([[0], [30], [49]])
        block_logits, thres_logits = policy(m, h, t, start)
        self.assertEqual(block_logits.shape, (3, 128))
        feasible = torch.isfinite(block_logits)
        self.assertEqual(feasible.sum(dim=-1).tolist(), [50, 20, 1])
        self.assertEqual(thres_logits.shape, (3, 1))

    def test_greedy_eval_drains(self):
        L = 50
        cands = list(range(1, 129))
        result = run(
            self.make_policy(cands), cands, B=2, L=L, sampling_mode="categorical-argmax"
        )
        self.assertFalse(bool((result.sequences[:, 3:] == MASK_ID).any()))


if __name__ == "__main__":
    unittest.main()
