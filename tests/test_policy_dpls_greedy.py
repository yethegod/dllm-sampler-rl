"""Tests for DPLS in the paper's fixed-block decode loop (remasking='policy').

The ablation control llada_8b_instruct_dit_confidence_BL32_alpha1_dpls.yaml trains the
paper's dit_confidence policy with sampling_mode 'dpls' and is evaluated with the
deterministic 'dpls-greedy', in the fixed-block loop and under AdaBlock. What has to
hold, none of which needs a GPU:
  1. Every forward that has candidates in a row's block commits >= 1 position of that
     row, so every block finishes inside its block_length budget, nothing is left
     masked, and NFE (steps_taken) counts only those forwards.
  2. 'dpls-greedy' is deterministic: the same output whatever the global RNG state.
  3. The same holds under adaptive_block, whose block lengths are not the trained 32.
  4. A sampling mode the policy loop does not implement fails up front.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.
Run with:  python tests/test_policy_dpls_greedy.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import generate_unified  # noqa: E402
from common.models.policy import DiTConfidencePolicy, PolicyHFWrapper  # noqa: E402

VOCAB = 40
MASK_ID = VOCAB - 1
DELIM = 7
L = 64
BLOCK = 16
PROMPT_L = 5


class StubDLLM:
    """Logits depend on (absolute position, forward index) only, never on the batch.

    Per-position sharpness spans ~uniform to near one-hot so the policy sees a spread
    of confidences; the mask token is never predicted.
    """

    def __init__(self, seed):
        self.seed = seed
        self.calls = 0
        self.dtype = torch.float32
        self.config = SimpleNamespace(hidden_size=8)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        g = torch.Generator().manual_seed(self.seed * 100003 + self.calls)
        self.calls += 1
        logits = torch.randn(S, VOCAB, generator=g) * (torch.rand(S, 1, generator=g) * 10.0)
        logits[:, MASK_ID] = -1e4
        return SimpleNamespace(logits=logits.unsqueeze(0).expand(B, S, VOCAB).clone())


def _policy(seed=0):
    torch.manual_seed(seed)
    core = DiTConfidencePolicy(
        hidden_dim=16,
        feedforward_dim=32,
        num_heads=2,
        time_embed_dim=16,
        confidences_top_p=1,
        smart_init=-2.0,
    )
    return PolicyHFWrapper(core, "dit_confidence").eval()


def _run(policy, mode, batch=3, rng_seed=0, **kw):
    torch.manual_seed(rng_seed)
    prompt = torch.randint(0, MASK_ID, (batch, PROMPT_L), generator=torch.Generator().manual_seed(1))
    return generate_unified(
        StubDLLM(3),
        prompt,
        remasking="policy",
        policy=policy,
        gen_length=L,
        block_length=BLOCK,
        mask_id=MASK_ID,
        sampling_mode=mode,
        full_context=True,
        record_unmask_order=True,
        **kw,
    )


class TestPolicyLoopDpls(unittest.TestCase):
    def setUp(self):
        self.policy = _policy()

    def _check_progress(self, res, mode):
        gen = res.sequences[:, PROMPT_L:]
        self.assertFalse(bool((gen == MASK_ID).any()), mode)
        order = res.unmask_order
        self.assertTrue(bool((order >= 0).all()), mode)
        for r in range(gen.shape[0]):
            nfe = int(res.steps_taken[r])
            self.assertLessEqual(nfe, L, mode)
            # Every counted forward committed something: the step ids used are exactly
            # 0..nfe-1, so no forward was spent without progress.
            self.assertEqual(sorted(set(order[r].tolist())), list(range(nfe)), mode)

    def test_every_forward_commits_and_sequence_completes(self):
        for mode in ("dpls", "dpls-greedy"):
            with torch.no_grad():
                res = _run(self.policy, mode)
            self._check_progress(res, mode)

    def test_blocks_fill_in_order(self):
        # A position of block k is never committed after one of block k+1.
        with torch.no_grad():
            res = _run(self.policy, "dpls-greedy")
        order = res.unmask_order.view(-1, L // BLOCK, BLOCK)
        self.assertTrue(bool((order.amax(-1)[:, :-1] < order.amin(-1)[:, 1:]).all()))

    def test_greedy_ignores_global_rng(self):
        with torch.no_grad():
            a = _run(self.policy, "dpls-greedy", rng_seed=0)
            b = _run(self.policy, "dpls-greedy", rng_seed=12345)
        self.assertTrue(torch.equal(a.sequences, b.sequences))
        self.assertTrue(torch.equal(a.unmask_order, b.unmask_order))
        self.assertTrue(torch.equal(a.steps_taken, b.steps_taken))

    def test_greedy_commits_more_than_one_somewhere(self):
        # With smart_init -2 over a 16-wide window DPLS takes several positions per
        # forward early in a block; one per forward would mean the greedy rule degenerated.
        with torch.no_grad():
            res = _run(self.policy, "dpls-greedy")
        self.assertTrue(bool((res.steps_taken < L).all()))

    def test_adaptive_block(self):
        with torch.no_grad():
            res = _run(
                self.policy,
                "dpls-greedy",
                batch=1,
                adaptive_block=True,
                delimiter_ids=(DELIM,),
                delimiter_threshold=0.05,
            )
        self._check_progress(res, "adaptive")
        self.assertEqual(sum(res.block_sizes), L)

    def test_unknown_mode_fails_up_front(self):
        with self.assertRaises(ValueError):
            _run(self.policy, "categorical")


if __name__ == "__main__":
    unittest.main()
