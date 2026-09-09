"""Tests for the v2 block-head fixes of remasking='block_unmask_policy'.

Job 3077216 trained the joint (block size, per-position unmask) policy for a full
epoch and its block head never moved: the block marginal stayed uniform, and every
block-head parameter ended within 0.016 of its init. Three things conspired, and
each has a fix with a property that can be checked without a GPU:

  1. boundary_proj was zero-initialised, so du/dh = 0 and the block term of the
     loss reached no trunk parameter (`boundary_init_gain`).
  2. The block head's own parameters are a 7-vector and a 128-vector at lr 3e-5,
     which under Adam cannot move more than ~lr per step (`policy_head_lr`: a
     separate optimizer group with its own base rate).
  3. The block log-prob was summed into the unmask log-prob of the same timestep
     and the row's loss averaged over its ~50 unmask steps, so its ~6 block
     decisions carried 1/T of the weight (`block_unmask_split_loss`: one clipped
     ratio per action type, each normalised by its own token count).

This repo has no pytest dependency, so the file is plain `unittest` and self-running.

Run with:  python tests/test_block_unmask_head_update.py
"""

import contextlib
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.models.policy import DiTBlockUnmaskPolicy, PolicyHFWrapper  # noqa: E402
from train.trainer import (  # noqa: E402
    BLOCK_HEAD_PARAM_NAMES,
    Trainer,
    _is_block_head_param,
)

CANDIDATES = (8, 16, 32)
POLICY_KW = dict(
    block_size_candidates=CANDIDATES,
    hidden_dim=16,
    feedforward_dim=32,
    num_heads=2,
    time_embed_dim=16,
    confidences_top_p=2,
    smart_init=-1.0,
)


def _policy(gain: float, window_cond: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    return DiTBlockUnmaskPolicy(
        window_cond=window_cond, boundary_init_gain=gain, **POLICY_KW
    )


def _inputs(B=2, L=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand((B, L), generator=g) < 0.6
    c = torch.rand((B, L, POLICY_KW["confidences_top_p"]), generator=g)
    t = torch.rand((B, 1), generator=g)
    start = torch.zeros((B, 1), dtype=torch.long)
    return m, c, t, start


# --------------------------------------------------------------------------------
# 1. boundary_init_gain
# --------------------------------------------------------------------------------


class TestBoundaryInit(unittest.TestCase):
    def test_gain_zero_is_the_legacy_zero_init(self):
        p = _policy(0.0)
        self.assertTrue((p.boundary_proj.weight == 0).all())
        self.assertTrue((p.boundary_proj.bias == 0).all())
        # Block logits are exactly the (zero) prior on every feasible candidate.
        logits = p.block_logits(*_inputs())
        self.assertTrue((logits[torch.isfinite(logits)] == 0).all())

    def test_gain_one_keeps_default_weight_and_zero_bias(self):
        torch.manual_seed(0)
        ref = nn.Linear(POLICY_KW["hidden_dim"], 1)  # what nn.Linear draws by default
        p = _policy(1.0)
        self.assertFalse((p.boundary_proj.weight == 0).all())
        self.assertTrue((p.boundary_proj.bias == 0).all())
        bound = 1 / POLICY_KW["hidden_dim"] ** 0.5
        self.assertLessEqual(p.boundary_proj.weight.abs().max().item(), bound + 1e-6)
        self.assertEqual(ref.weight.shape, p.boundary_proj.weight.shape)

    def test_gain_scales_the_weight(self):
        torch.manual_seed(0)
        a = _policy(1.0, seed=3).boundary_proj.weight
        b = _policy(0.5, seed=3).boundary_proj.weight
        torch.testing.assert_close(b, 0.5 * a)

    def test_negative_gain_rejected(self):
        with self.assertRaises(ValueError):
            _policy(-1.0)

    def test_infeasible_candidates_still_masked(self):
        p = _policy(1.0)
        m, c, t, start = _inputs(B=1, L=32)
        start = torch.tensor([[24]])  # only b=8 fits
        logits = p.block_logits(m, c, t, start)
        self.assertTrue(torch.isfinite(logits[0, 0]))
        self.assertTrue(torch.isinf(logits[0, 1:]).all())

    def _trunk_grad_from_block_head(self, gain):
        p = _policy(gain)
        m, c, t, start = _inputs()
        logits = p.block_logits(m, c, t, start)
        # A block-only loss: log-prob of candidate 0 under the categorical.
        loss = -torch.log_softmax(logits, dim=-1)[:, 0].sum()
        loss.backward()
        trunk = p.confidence_proj.weight.grad
        head = p.boundary_proj.weight.grad
        return trunk, head, p.block_size_bias.grad

    def test_zero_init_cuts_the_trunk_gradient(self):
        trunk, head, bias = self._trunk_grad_from_block_head(0.0)
        # This is the frozen-head mechanism of job 3077216: the head's own
        # parameters see a gradient, the shared trunk sees none.
        self.assertTrue((trunk == 0).all())
        self.assertFalse((head == 0).all())
        self.assertFalse((bias == 0).all())

    def test_nonzero_init_reaches_the_trunk(self):
        trunk, head, bias = self._trunk_grad_from_block_head(1.0)
        self.assertGreater(trunk.abs().max().item(), 0.0)
        self.assertGreater(head.abs().max().item(), 0.0)

    def test_window_cond_still_starts_identical_to_independent(self):
        # boundary_init_gain and window_cond are orthogonal: the window embedding
        # is still zero, so the two factorisations agree at init for any gain.
        cond = _policy(1.0, window_cond=True, seed=5)
        indep = _policy(1.0, window_cond=False, seed=5)
        indep.load_state_dict(cond.state_dict(), strict=False)
        m, c, t, start = _inputs()
        end = start + 16
        u_c, b_c = cond(m, c, t, start, end)
        u_i, b_i = indep(m, c, t, start)
        torch.testing.assert_close(u_c, u_i)
        torch.testing.assert_close(b_c, b_i)


# --------------------------------------------------------------------------------
# 2. policy_head_lr: optimizer groups
# --------------------------------------------------------------------------------


class TestHeadParamNames(unittest.TestCase):
    def test_matches_through_wrapper_and_ddp_prefixes(self):
        for name in (
            "block_size_bias",
            "base_policy.block_size_bias",
            "module.base_policy.boundary_proj.weight",
            "base_policy.boundary_proj.bias",
        ):
            self.assertTrue(_is_block_head_param(name), name)

    def test_does_not_match_the_unmask_head_or_trunk(self):
        for name in (
            # Read by the unmask head only; a head-lr group must not touch it.
            "base_policy.window_embedding.weight",
            "base_policy.output_proj.weight",
            "base_policy.output_proj.bias",
            "base_policy.confidence_proj.weight",
            "base_policy.mask_embedding.weight",
            "base_policy.transformer_blocks.0.ada_conditioning.weight",
            "base_policy.final_norm.weight",
        ):
            self.assertFalse(_is_block_head_param(name), name)

    def test_every_head_name_exists_on_the_policy(self):
        p = PolicyHFWrapper(_policy(1.0, window_cond=True), "dit_block_unmask")
        names = {n for n, _ in p.named_parameters()}
        for head in BLOCK_HEAD_PARAM_NAMES:
            self.assertTrue(
                any(head in n.split(".") for n in names), f"{head} not on the policy"
            )


class _OptimizerStub:
    """Just enough of Trainer for create_optimizer: args, model, and HF's helpers."""

    create_optimizer = Trainer.create_optimizer
    get_decay_parameter_names = Trainer.get_decay_parameter_names
    get_optimizer_cls_and_kwargs = staticmethod(Trainer.get_optimizer_cls_and_kwargs)

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = None


def _training_args(tmpdir, **overrides):
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=tmpdir, learning_rate=3e-5, weight_decay=0.1, report_to=[]
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestHeadLearningRateGroups(unittest.TestCase):
    def test_groups_split_head_from_the_rest(self):
        model = PolicyHFWrapper(_policy(1.0, window_cond=True), "dit_block_unmask")
        with tempfile.TemporaryDirectory() as tmp:
            args = _training_args(tmp, policy_head_lr=1e-3)
            opt = _OptimizerStub(model, args).create_optimizer()

        by_param = {}
        for group in opt.param_groups:
            for p in group["params"]:
                by_param[id(p)] = (group["lr"], group["weight_decay"])
        all_params = dict(model.named_parameters())
        # Every parameter is in exactly one group.
        self.assertEqual(len(by_param), len(all_params))
        self.assertEqual(sum(len(g["params"]) for g in opt.param_groups), len(all_params))

        for name, p in all_params.items():
            lr, wd = by_param[id(p)]
            if _is_block_head_param(name):
                self.assertEqual(lr, 1e-3, name)
            else:
                self.assertEqual(lr, 3e-5, name)
            # HF's decay rule is preserved inside the head group too: the prior
            # logits are a bias and must not be pulled back toward uniform.
            if name.endswith("block_size_bias") or name.endswith("bias"):
                self.assertEqual(wd, 0.0, name)
        # Trainer logs param_groups[0]["lr"] as the learning rate: keep it the base.
        self.assertEqual(opt.param_groups[0]["lr"], 3e-5)

    def test_refuses_a_policy_without_a_block_head(self):
        from common.models.policy import DiTConfidencePolicy

        torch.manual_seed(0)
        model = PolicyHFWrapper(
            DiTConfidencePolicy(
                **{k: v for k, v in POLICY_KW.items() if k != "block_size_candidates"}
            ),
            "dit_confidence",
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = _training_args(tmp, policy_head_lr=1e-3)
            with self.assertRaises(AssertionError):
                _OptimizerStub(model, args).create_optimizer()


# --------------------------------------------------------------------------------
# 3. block_unmask_split_loss
# --------------------------------------------------------------------------------

L = 4  # positions
K = 3  # block candidates
T = 10  # timesteps, all active
DECISION_STEPS = (0, 5)  # the block head acts twice per row


class _TinyTwoHead(nn.Module):
    """(B,T,L) -> ((B,T,L) unmask logits, (B,T,K) block logits), one param each."""

    def __init__(self):
        super().__init__()
        self.pos_scale = nn.Parameter(torch.tensor([0.5, -0.5, 1.0, -1.0]))
        self.block_bias = nn.Parameter(torch.tensor([0.2, 0.0, -0.2]))

    def forward(self, x):
        B, T_, _ = x.shape
        return x * self.pos_scale, self.block_bias.expand(B, T_, K)


class _StubArgs:
    def __init__(self, split, coef=1.0, timestep_batch_size=None):
        self.fp16 = False
        self.remasking = "block_unmask_policy"
        self.sampling_mode = "bernoulli"
        self.block_unmask_split_loss = split
        self.block_loss_coef = coef
        self.loglikelihood_dtype = torch.float32
        self.epsilon = 0.2
        self.timestep_batch_size = timestep_batch_size
        self.es_thresholds = None
        self.dpls_stop_logit = 0.0


class _StubAccelerator:
    def backward(self, loss):
        loss.backward()

    def no_sync(self, model):
        return contextlib.nullcontext()

    def gather_for_metrics(self, tensor):
        return tensor


class _StubTrainer:
    compute_loss = Trainer.compute_loss
    _get_per_timestep_logps_block = Trainer._get_per_timestep_logps_block
    _clipped_surrogate = Trainer._clipped_surrogate
    _block_unmask_loglik_parts = Trainer._block_unmask_loglik_parts
    _block_unmask_joint_loglik = Trainer._block_unmask_joint_loglik

    def __init__(self, split, coef=1.0, timestep_batch_size=None):
        self.args = _StubArgs(split, coef, timestep_batch_size)
        self.accelerator = _StubAccelerator()
        self.beta = 0.0
        self._metrics = {"train": defaultdict(list)}


def _rollout(model, B=4, seed=0):
    """One batch in the block_unmask layout, with old log-probs from `model` itself
    so the GRPO ratio is exactly 1 on both heads (as on the first inner iteration)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((B, T, L), generator=g)
    samples_pos = (torch.rand((B, T, L), generator=g) < 0.5).long()
    samples_blk = torch.randint(0, K, (B, T), generator=g)
    masks = torch.ones((B, T, L + 1), dtype=torch.bool)
    masks[..., L] = False
    for t in DECISION_STEPS:
        masks[:, t, L] = True
    samples = torch.cat([samples_pos, samples_blk.unsqueeze(-1)], dim=-1)
    with torch.no_grad():
        u, b = model(x)
        logits = torch.cat([u, b], dim=-1)
    stub = _StubTrainer(split=False)
    pos_ll, block_ll = stub._block_unmask_loglik_parts(samples, logits, masks)
    return {
        "samples": samples,
        "sampling_masks": masks,
        "policy_inputs": [x],
        "old_per_timestep_logps": pos_ll + block_ll,
        "old_pos_logps": pos_ll,
        "old_block_logps": block_ll,
        "state_history": None,
    }


def _grads(model, trainer, policy_output, advantages):
    model.zero_grad(set_to_none=True)
    loss = trainer.compute_loss(
        model, {"policy_outputs": [policy_output], "advantages": advantages}
    )
    trainer.accelerator.backward(loss)
    return loss.detach(), model.pos_scale.grad.clone(), model.block_bias.grad.clone()


class TestSplitLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _TinyTwoHead()
        self.out = _rollout(self.model)
        # Group-normalised advantages: zero mean, so a ratio-1 loss is exactly 0.
        self.adv = torch.tensor([1.0, -1.0, 0.5, -0.5])

    def test_loss_is_zero_at_ratio_one_in_both_modes(self):
        for split in (False, True):
            loss, _, _ = _grads(self.model, _StubTrainer(split), self.out, self.adv)
            self.assertAlmostEqual(loss.item(), 0.0, places=6, msg=f"{split=}")

    def test_unmask_gradient_unchanged_block_gradient_renormalised(self):
        _, pos_joint, blk_joint = _grads(
            self.model, _StubTrainer(split=False), self.out, self.adv
        )
        _, pos_split, blk_split = _grads(
            self.model, _StubTrainer(split=True), self.out, self.adv
        )
        # The unmask head sees the same objective either way.
        torch.testing.assert_close(pos_split, pos_joint)
        # At ratio 1 the block gradient is A * dlogp/dtheta per decision, divided
        # by T active steps (joint) or by the row's decision count (split).
        expected_factor = T / len(DECISION_STEPS)
        self.assertGreater(blk_joint.abs().max().item(), 0.0)
        torch.testing.assert_close(blk_split, expected_factor * blk_joint)

    def test_block_loss_coef_scales_only_the_block_term(self):
        _, pos_1, blk_1 = _grads(
            self.model, _StubTrainer(split=True, coef=1.0), self.out, self.adv
        )
        _, pos_2, blk_2 = _grads(
            self.model, _StubTrainer(split=True, coef=0.25), self.out, self.adv
        )
        torch.testing.assert_close(pos_2, pos_1)
        torch.testing.assert_close(blk_2, 0.25 * blk_1)

    def test_chunked_matches_unchunked_under_split(self):
        _, pos_a, blk_a = _grads(
            self.model, _StubTrainer(split=True), self.out, self.adv
        )
        _, pos_b, blk_b = _grads(
            self.model, _StubTrainer(split=True, timestep_batch_size=3), self.out, self.adv
        )
        torch.testing.assert_close(pos_b, pos_a)
        torch.testing.assert_close(blk_b, blk_a)

    def test_entropy_still_reported_per_head(self):
        trainer = _StubTrainer(split=True)
        _grads(self.model, trainer, self.out, self.adv)
        self.assertEqual(len(trainer._metrics["train"]["entropy"]), 1)
        self.assertEqual(len(trainer._metrics["train"]["block_entropy"]), 1)
        self.assertGreater(trainer._metrics["train"]["block_entropy"][0], 0.0)

    def test_block_entropy_independent_of_chunking(self):
        # Chunks without a block decision used to contribute a 0 mean at equal
        # weight; with timestep_batch_size=3 and decisions at t=0 and t=5 only
        # 2 of 4 chunks decide, which halved the logged value.
        vals = {}
        for tbs in (None, 3, 1):
            trainer = _StubTrainer(split=True, timestep_batch_size=tbs)
            _grads(self.model, trainer, self.out, self.adv)
            vals[tbs] = (
                trainer._metrics["train"]["block_entropy"][0],
                trainer._metrics["train"]["entropy"][0],
            )
        for tbs in (3, 1):
            self.assertAlmostEqual(vals[tbs][0], vals[None][0], places=5)
            self.assertAlmostEqual(vals[tbs][1], vals[None][1], places=5)
        # And it is the entropy of the block head's own distribution.
        logits = self.model.block_bias.detach()
        p = torch.softmax(logits, -1)
        self.assertAlmostEqual(vals[None][0], float(-(p * p.log()).sum()), places=5)


if __name__ == "__main__":
    unittest.main()
