"""Tests for DPLS in the block_unmask_policy decode loop (sampling_mode 'dpls' and the
deterministic eval counterpart 'dpls-greedy').

Why DPLS here: training-mode Bernoulli can draw no position at all, and the per-row
block loop only advances a block once it is fully unmasked, under one global
max_steps cap, so rows that pick small blocks run out of forwards before writing the
answer. DPLS always commits at least one position of a row that has any. What has to
hold, none of which needs a GPU:
  1. The sampler and dpls_batch_loglik agree (the likelihood normalises over every
     ordered output the sampler can produce, and matches its frequencies).
  2. dpls_greedy is deterministic, commits the top-utility candidates, at least one per
     row with candidates, and follows DPLS's collective stopping rather than the
     tau -> 0 limit.
  3. In the decode loop every step with candidates commits >= 1 position, the sequence
     always completes, the record holds ordered indices inside the row's window, and
     replaying `policy(*policy_inputs)` through the trainer's log-lik reproduces the
     rollout's (ratio 1 on the first inner iteration).

This repo has no pytest dependency, so the file is plain `unittest` and self-running.
Run with:  python tests/test_block_unmask_dpls.py
"""

import functools
import itertools
import math
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import (  # noqa: E402 - needs the sys.path line above
    _block_unmask_policy_loop,
)
from common.generation.sampling import (  # noqa: E402
    bernoulli_batch_loglik,
    dpls_batch_loglik,
    dpls_greedy,
    dpls_sample,
)
from common.models.policy import DiTBlockUnmaskPolicy  # noqa: E402
from train.trainer import Trainer  # noqa: E402

L = 32
P = 1  # confidences_top_p
CANDIDATES = (4, 8, 16)
MASK_ID = 0
POLICY_KW = dict(
    block_size_candidates=CANDIDATES,
    window_cond=True,
    boundary_init_gain=1.0,
    hidden_dim=16,
    feedforward_dim=32,
    num_heads=2,
    time_embed_dim=16,
    confidences_top_p=P,
    smart_init=-2.0,
)


def _pad(seq, n):
    return torch.tensor([list(seq) + [-1] * (n - len(seq))])


def _as_set(samples: torch.Tensor) -> torch.Tensor:
    """Ordered indices padded with -1 -> bool set. scatter_add, not scatter: the padding
    is clamped onto index 0 and must not overwrite a real pick of position 0."""
    return (
        torch.zeros_like(samples, dtype=torch.int)
        .scatter_add(-1, samples.clamp(min=0), (samples >= 0).int())
        .bool()
    )


class TestDplsSampler(unittest.TestCase):
    utilities = torch.tensor([[0.3, -1.2, 0.8, 2.0]])
    mask = torch.tensor([[True, True, False, True]])

    def _outputs(self):
        avail = [i for i in range(4) if self.mask[0, i]]
        return [s for k in range(1, len(avail) + 1) for s in itertools.permutations(avail, k)]

    def test_loglik_normalises_over_all_ordered_outputs(self):
        total = sum(
            math.exp(dpls_batch_loglik(_pad(s, 4), self.utilities, 0.0, self.mask).item())
            for s in self._outputs()
        )
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_sampler_frequencies_match_loglik(self):
        torch.manual_seed(0)
        n = 20000
        samples, _ = dpls_sample(self.utilities.repeat(n, 1), 0.0, self.mask.repeat(n, 1))
        counts = Counter(tuple(x for x in row if x >= 0) for row in samples.tolist())
        for s in self._outputs():
            p = math.exp(dpls_batch_loglik(_pad(s, 4), self.utilities, 0.0, self.mask).item())
            self.assertLess(abs(counts[s] / n - p), 0.015, s)
        self.assertTrue(all(2 not in s for s in counts))  # masked-out position never drawn


class TestDplsGreedy(unittest.TestCase):
    def test_deterministic_and_takes_top_utilities(self):
        torch.manual_seed(1)
        u = torch.randn(64, 32) * 1.5 - 1.5
        m = torch.rand(64, 32) < 0.7
        s1, c1 = dpls_greedy(u, 0.0, m)
        s2, c2 = dpls_greedy(u, 0.0, m)
        self.assertTrue(torch.equal(s1, s2) and torch.equal(c1, c2))
        self.assertTrue(torch.equal(_as_set(s1), c1))
        for i in range(64):
            k = int(c1[i].sum())
            top = u[i].masked_fill(~m[i], -1e9).topk(k).indices
            self.assertEqual(set(top.tolist()), set(s1[i, :k].tolist()))
            self.assertTrue(bool((s1[i, k:] == -1).all()))

    def test_at_least_one_and_empty_rows(self):
        u = torch.randn(3, 8) - 5.0  # all utilities far below STOP
        m = torch.tensor([[True] * 8, [False] * 8, [False] * 7 + [True]])
        s, c = dpls_greedy(u, 0.0, m)
        self.assertEqual(c.sum(-1).tolist(), [1, 0, 1])
        self.assertTrue(bool((s[1] == -1).all()))

    def test_follows_collective_stopping_not_tau0_limit(self):
        # 32 equal candidates at u=-2: the tau->0 rule (u > stop) commits 1, DPLS ~4.7.
        u = torch.full((1, 32), -2.0)
        m = torch.ones_like(u, dtype=torch.bool)
        self.assertEqual(int(dpls_greedy(u, 0.0, m)[1].sum()), 4)
        torch.manual_seed(2)
        mean = dpls_sample(u.repeat(4000, 1), 0.0, m.repeat(4000, 1))[1].sum(-1).float().mean()
        self.assertLess(abs(mean.item() - 4.0), 1.0)
        # a small window stops at the forced first pick
        self.assertEqual(int(dpls_greedy(u[:, :8], 0.0, m[:, :8])[1].sum()), 1)


def _run_loop(policy, mode, B=4, seed=3):
    torch.manual_seed(seed)
    prompt_L, V = 3, 6
    x = torch.full((B, prompt_L + L), MASK_ID, dtype=torch.long)
    x[:, :prompt_L] = 1
    steps_taken = torch.zeros(B, dtype=torch.int)
    g = torch.Generator().manual_seed(seed)

    def forward_logits():
        logits = torch.randn((B, L, V), generator=g)
        logits[..., MASK_ID] = -1e9  # never predict the mask token
        probs = torch.softmax(logits, dim=-1)
        return None, probs, probs.argmax(dim=-1)

    rec = _block_unmask_policy_loop(
        x, prompt_L, L, MASK_ID, L, steps_taken, policy, forward_logits, None,
        CANDIDATES, mode, "categorical", P, 1.0, dpls_stop_logit=0.0,
    )
    return x[:, prompt_L:], steps_taken, rec


def _stub_trainer(sampling_mode):
    args = SimpleNamespace(
        fp16=False,
        remasking="block_unmask_policy",
        block_unmask_split_loss=True,
        sampling_mode=sampling_mode,
        dpls_stop_logit=0.0,
        loglikelihood_dtype=torch.float32,
    )
    stub = SimpleNamespace(args=args)
    stub._block_unmask_loglik_parts = functools.partial(
        Trainer._block_unmask_loglik_parts, stub
    )
    return stub


class TestLoopWithDpls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = DiTBlockUnmaskPolicy(**POLICY_KW).eval()

    def test_every_step_commits_and_sequence_completes(self):
        for mode in ("dpls", "dpls-greedy"):
            with torch.no_grad():
                gen, steps, rec = _run_loop(self.policy, mode)
            self.assertTrue(bool((gen != MASK_ID).all()), mode)
            self.assertTrue(bool((steps <= L).all()), mode)
            cand = rec["sampling_masks"][..., :L]
            active = cand.any(-1)
            picks = _as_set(rec["samples"][..., :L])
            self.assertTrue(bool((picks.sum(-1)[active] >= 1).all()), mode)
            self.assertFalse(bool((picks & ~cand).any()), mode)  # only in-window candidates
            self.assertTrue(bool((rec["samples"][..., :L][~active] == -1).all()), mode)
            # every generated position was committed exactly once
            self.assertEqual(int(picks.sum()), gen.numel(), mode)
            self.assertTrue(torch.equal(rec["productive_steps"], steps), mode)

    def test_replay_reproduces_rollout_loglik(self):
        with torch.no_grad():
            _, _, rec = _run_loop(self.policy, "dpls")
            fresh = torch.cat(self.policy(*rec["policy_inputs"]), dim=-1)
        stub = _stub_trainer("dpls")
        rec_pos, rec_blk = stub._block_unmask_loglik_parts(
            rec["samples"], rec["sampling_inputs"], rec["sampling_masks"]
        )
        new_pos, new_blk = stub._block_unmask_loglik_parts(
            rec["samples"], fresh, rec["sampling_masks"]
        )
        torch.testing.assert_close(new_pos, rec_pos)
        torch.testing.assert_close(new_blk, rec_blk)
        active = rec["sampling_masks"][..., :L].any(-1)
        self.assertTrue(bool((rec_pos[~active] == 0).all()))
        self.assertTrue(bool(torch.isfinite(rec_pos).all()))

    def test_greedy_is_reproducible(self):
        with torch.no_grad():
            a = _run_loop(self.policy, "dpls-greedy", seed=9)
            b = _run_loop(self.policy, "dpls-greedy", seed=9)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[2]["samples"], b[2]["samples"]))

    def test_bernoulli_record_is_still_the_draw(self):
        with torch.no_grad():
            _, _, rec = _run_loop(self.policy, "bernoulli")
        pos = rec["samples"][..., :L]
        self.assertTrue(bool(((pos == 0) | (pos == 1)).all()))


class TestTrainerDispatch(unittest.TestCase):
    def test_loglik_parts_follow_sampling_mode(self):
        torch.manual_seed(4)
        B, T, K = 2, 5, len(CANDIDATES)
        logits = torch.randn(B, T, L + K)
        masks = torch.zeros(B, T, L + 1, dtype=torch.bool)
        masks[..., 4:12] = True
        u = logits[..., :L]
        seqs, sets = dpls_sample(u.reshape(-1, L), 0.0, masks[..., :L].reshape(-1, L))
        samples = torch.cat([seqs.view(B, T, L), torch.zeros(B, T, 1, dtype=torch.long)], -1)
        pos, _ = _stub_trainer("dpls")._block_unmask_loglik_parts(samples, logits, masks)
        torch.testing.assert_close(pos, dpls_batch_loglik(seqs.view(B, T, L), u, 0.0, masks[..., :L]))
        draws = torch.cat([sets.view(B, T, L).long(), torch.zeros(B, T, 1, dtype=torch.long)], -1)
        pos_b, _ = _stub_trainer("bernoulli")._block_unmask_loglik_parts(draws, logits, masks)
        torch.testing.assert_close(pos_b, bernoulli_batch_loglik(draws[..., :L], u, mask_index=masks[..., :L]))

    def test_dpls_entropy_and_gradients_are_finite(self):
        torch.manual_seed(0)
        policy = DiTBlockUnmaskPolicy(**POLICY_KW)
        with torch.no_grad():
            _, _, rec = _run_loop(policy.eval(), "dpls")
        policy.train()
        stub = _stub_trainer("dpls")
        (pos, blk), ent = Trainer._get_per_timestep_logps_block(
            stub, policy, rec["samples"], rec["sampling_masks"], rec["policy_inputs"],
            sampling_mode="dpls", return_entropy=True,
        )
        ent = ent.detach()  # compute_loss detaches it too; it is logged, not optimised
        self.assertTrue(bool(torch.isfinite(ent).all()))
        self.assertGreater(float(ent[1]), 0)  # counted steps with candidates
        (-(pos.sum() + blk.sum())).backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(grads and all(bool(torch.isfinite(g).all()) for g in grads))


if __name__ == "__main__":
    unittest.main()
