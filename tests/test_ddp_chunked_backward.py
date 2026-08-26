"""Regression tests for the DDP collective structure of `train.trainer.Trainer`.

Two things in `compute_loss`/`training_step` are invisible on a single process and are
exactly the things that deadlock a real 8-GPU run:

  1. `compute_loss` backwards every timestep chunk but the last. The number of chunks
     is a function of T, which varies per rank with the rollout length, so a per-chunk
     collective mismatches across ranks. Every non-final chunk must therefore run its
     *forward as well as* its backward under `no_sync` (DDP arms reduction inside
     forward, not backward), leaving exactly one synchronising backward per rank.
  2. The "all advantages are zero" early return in `training_step`. `advantages` is a
     per-rank slice, so one rank skipping while another enters backward is a hang.

Neither needs a GPU to reproduce: gloo on CPU has the same collective semantics. These
tests run two real ranks with a tiny policy stand-in, under a watchdog -- the failure
mode under test is a hang, so a timeout is a test failure, not an infrastructure hiccup.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.

Run with:  python tests/test_ddp_chunked_backward.py
"""

import contextlib
import os
import queue as queue_lib
import socket
import sys
import time
import traceback
import unittest
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root. The
# spawned ranks re-execute this module, so they pick the same path up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.trainer import Trainer  # noqa: E402 - needs the sys.path line above

# Generous enough that a slow login node never trips it, short enough that a genuine
# deadlock does not hold the suite for unittest's default (no timeout at all).
WATCHDOG_SECONDS = 240
WORLD_SIZE = 2
BL = 5  # "block length": positions the policy scores per timestep


# --------------------------------------------------------------------------------
# Minimal stand-ins for the pieces of the real training stack the loss touches
# --------------------------------------------------------------------------------


class _TinyPolicy(nn.Module):
    """Elementwise affine map (B, T, BL) -> (B, T, BL).

    Small, but every parameter participates in every forward, which is what DDP
    requires and what makes the bucket count stable enough to assert on.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.scale = nn.Parameter(torch.randn(BL, generator=gen))
        self.bias = nn.Parameter(torch.randn(BL, generator=gen))

    def forward(self, x):
        return x * self.scale + self.bias


class _StubArgs:
    """The `self.args` fields `compute_loss` and `_get_per_timestep_logps_block` read."""

    def __init__(self, timestep_batch_size):
        self.fp16 = False
        self.sampling_mode = "bernoulli"
        self.loglikelihood_dtype = torch.float32
        self.epsilon = 0.2
        self.timestep_batch_size = timestep_batch_size
        self.es_thresholds = None
        self.dpls_stop_logit = 0.0


class _StubAccelerator:
    """The three `self.accelerator` entry points the code under test uses.

    `no_sync` and `backward` are forwarded to torch verbatim so the DDP semantics under
    test are the real ones; only the accelerate bookkeeping around them is dropped.
    """

    def __init__(self):
        self.backward_calls = 0

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def no_sync(self, model):
        if isinstance(model, DistributedDataParallel):
            return model.no_sync()
        return contextlib.nullcontext()

    @staticmethod
    def _all_gather(tensor):
        flat = tensor.reshape(-1).contiguous()
        out = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
        dist.all_gather(out, flat)
        return torch.cat(out)

    def gather(self, tensor):
        return self._all_gather(tensor)

    def gather_for_metrics(self, tensor):
        return self._all_gather(tensor)


class _StubTrainer:
    """Carries the real methods under test, bound to stubbed collaborators.

    Borrowing the unbound functions (rather than reimplementing them) is the point:
    the test fails when `train/trainer.py` changes, which is what it is here to guard.
    """

    compute_loss = Trainer.compute_loss
    _get_per_timestep_logps_block = Trainer._get_per_timestep_logps_block
    _skip_step_globally = Trainer._skip_step_globally

    def __init__(self, timestep_batch_size):
        self.args = _StubArgs(timestep_batch_size)
        self.accelerator = _StubAccelerator()
        self.beta = 0.0
        self._metrics = {"train": defaultdict(list)}


def _make_policy_output(batch_size, num_timesteps, seed):
    """One group-batch of rollout data in the layout `compute_loss` expects."""
    gen = torch.Generator().manual_seed(seed)
    shape = (batch_size, num_timesteps, BL)
    return {
        "sampling_masks": torch.ones(shape, dtype=torch.bool),
        "samples": torch.rand(shape, generator=gen) < 0.5,
        "policy_inputs": [torch.randn(shape, generator=gen)],
        # Centred on the log-likelihood the tiny policy actually produces (BL roughly
        # fair Bernoullis, so ~BL*log(0.5)) with a little jitter, so the GRPO ratio
        # lands near 1. Ratios far outside the clip range would put every element on
        # the constant branch of torch.min, where no gradient flows, and the gradient
        # comparisons below would pass on all-zero tensors.
        "old_per_timestep_logps": BL * torch.log(torch.tensor(0.5))
        + 0.1 * torch.randn(batch_size, num_timesteps, generator=gen),
        "state_history": None,
    }


def _grads(model):
    return [p.grad.detach().clone() for p in model.parameters()]


def _run_one_step(trainer, model, policy_outputs, advantages):
    """A full training step: chunked backwards inside, final backward outside."""
    model.zero_grad(set_to_none=False)
    loss = trainer.compute_loss(
        model, {"policy_outputs": policy_outputs, "advantages": advantages}
    )
    trainer.accelerator.backward(loss)
    return loss


# --------------------------------------------------------------------------------
# Distributed harness
# --------------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entrypoint(worker, rank, world_size, port, queue):
    """Child-process wrapper: set up gloo, run `worker`, report failures by value.

    Exceptions are shipped as text because the parent cannot see a child's traceback,
    and because a rank that dies mid-collective would otherwise surface only as the
    *other* rank's timeout.
    """
    try:
        torch.set_num_threads(1)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, init_method="env://"
        )
        worker(rank, world_size)
        # No closing barrier on purpose: a rank that fails an assertion never reaches
        # one, and the healthy rank would then turn a clear failure into a timeout.
        queue.put((rank, None))
    except BaseException:  # noqa: BLE001 - the parent turns this back into a failure
        queue.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(worker, world_size=WORLD_SIZE, timeout=WATCHDOG_SECONDS):
    """Run `worker` on `world_size` gloo ranks, failing the test on hang or error."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_entrypoint, args=(worker, rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()

    deadline = time.monotonic() + timeout
    try:
        # Drain before joining: a child blocked writing to a full queue pipe would
        # otherwise never exit, and a reporting problem would masquerade as the hang
        # this test is supposed to detect.
        reports = {}
        while len(reports) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rank, err = queue.get(timeout=min(remaining, 5.0))
            except queue_lib.Empty:
                if not any(p.is_alive() for p in procs):
                    break  # everyone exited; nothing more is coming
                continue
            reports[rank] = err

        for proc in procs:
            proc.join(max(0.0, deadline - time.monotonic()))

        # A real assertion failure is more informative than the timeout it may have
        # induced on the peer rank, so report it first.
        failures = [f"rank {r}:\n{e}" for r, e in sorted(reports.items()) if e]
        if failures:
            raise AssertionError("\n\n".join(failures))

        hung = [i for i, p in enumerate(procs) if p.is_alive()]
        if hung:
            raise AssertionError(
                f"ranks {hung} still alive after {timeout}s -- the collective "
                f"structure deadlocked (this is the failure mode under test)"
            )

        missing = sorted(set(range(world_size)) - set(reports))
        assert not missing, (
            f"ranks {missing} exited without reporting; "
            f"exit codes {[p.exitcode for p in procs]}"
        )
        bad = [(i, p.exitcode) for i, p in enumerate(procs) if p.exitcode != 0]
        assert not bad, f"non-zero exit codes: {bad}"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
                proc.join(5)


# --------------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------------

# Deliberately not a multiple of the chunk size, and different per rank: rank 0 gets
# ceil(6/4)=2 chunks, rank 1 gets ceil(15/4)=4. Anything that syncs per chunk mismatches.
_UNEQUAL_T = {0: 6, 1: 15}
_CHUNK_BS = 4


def _worker_unequal_chunks(rank, world_size):
    """Unequal chunk counts per rank must still yield exactly one reduction each."""
    model = DistributedDataParallel(_TinyPolicy(seed=0))
    trainer = _StubTrainer(timestep_batch_size=_CHUNK_BS)

    reductions = [0]

    def counting_hook(state, bucket):
        reductions[0] += 1
        fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
        return fut.then(lambda f: f.value()[0].div_(world_size))

    model.register_comm_hook(state=None, hook=counting_hook)

    policy_outputs = [_make_policy_output(2, _UNEQUAL_T[rank], seed=100 + rank)]
    advantages = torch.tensor([1.0, -1.0])

    # Warm-up step: DDP rebuilds its buckets after the first backward, so the bucket
    # count is only stable from the second step on. Also establishes the reference.
    _run_one_step(trainer, model, policy_outputs, advantages)

    reductions[0] = 0
    _run_one_step(trainer, model, policy_outputs, advantages)
    buckets_per_sync = reductions[0]
    assert buckets_per_sync > 0, "reference step reduced nothing; hook never fired"

    n_chunks = -(-_UNEQUAL_T[rank] // _CHUNK_BS)
    assert trainer.accelerator.backward_calls == 2 * n_chunks, (
        f"rank {rank} expected {n_chunks} backwards per step "
        f"(got {trainer.accelerator.backward_calls} over two steps)"
    )

    # The real assertion: however many chunks this rank had, the step reduced exactly
    # as much as a single synchronised backward does -- not once per chunk.
    reductions[0] = 0
    _run_one_step(trainer, model, policy_outputs, advantages)
    assert reductions[0] == buckets_per_sync, (
        f"rank {rank} ({n_chunks} chunks) triggered {reductions[0]} bucket reductions, "
        f"expected {buckets_per_sync} (one synchronised backward)"
    )


def _worker_grads_equal_across_ranks(rank, world_size):
    """After the single synchronised backward, both ranks hold identical gradients."""
    model = DistributedDataParallel(_TinyPolicy(seed=0))
    trainer = _StubTrainer(timestep_batch_size=_CHUNK_BS)

    # Different data *and* different chunk counts per rank, so identical gradients can
    # only come from a reduction that actually happened.
    policy_outputs = [_make_policy_output(2, _UNEQUAL_T[rank], seed=200 + rank)]
    advantages = torch.tensor([0.5, -1.5]) * (rank + 1)

    _run_one_step(trainer, model, policy_outputs, advantages)
    mine = _grads(model)

    gathered = []
    for grad in mine:
        buf = [torch.empty_like(grad) for _ in range(world_size)]
        dist.all_gather(buf, grad.contiguous())
        gathered.append(buf)

    for idx, (grad, buf) in enumerate(zip(mine, gathered)):
        assert torch.allclose(buf[0], buf[1], atol=1e-6), (
            f"param {idx} differs across ranks after the synchronised backward:\n"
            f"{buf[0]}\nvs\n{buf[1]}"
        )
        assert torch.isfinite(grad).all(), f"param {idx} gradient is not finite"
        assert grad.abs().max() > 0, (
            f"param {idx} gradient is all-zero; the test would pass vacuously"
        )


def _worker_chunked_matches_unchunked(rank, world_size):
    """Chunking the backward must not change the gradient it produces."""
    policy_outputs = [_make_policy_output(2, _UNEQUAL_T[rank], seed=300 + rank)]
    advantages = torch.tensor([1.25, -0.75])

    grads_by_mode = {}
    losses_by_mode = {}
    for label, timestep_bs in (("unchunked", None), ("chunked", _CHUNK_BS)):
        model = DistributedDataParallel(_TinyPolicy(seed=0))
        trainer = _StubTrainer(timestep_batch_size=timestep_bs)
        loss = _run_one_step(trainer, model, policy_outputs, advantages)
        grads_by_mode[label] = _grads(model)
        losses_by_mode[label] = loss.detach().clone()

    n_chunks = -(-_UNEQUAL_T[rank] // _CHUNK_BS)
    assert n_chunks > 1, "test data must produce more than one chunk to be meaningful"

    assert torch.allclose(
        losses_by_mode["unchunked"], losses_by_mode["chunked"], atol=1e-5
    ), (
        f"rank {rank} loss differs: {losses_by_mode['unchunked']} "
        f"vs {losses_by_mode['chunked']}"
    )
    for idx, (ref, got) in enumerate(
        zip(grads_by_mode["unchunked"], grads_by_mode["chunked"])
    ):
        assert torch.allclose(ref, got, atol=1e-5), (
            f"rank {rank} param {idx} gradient changed with chunking:\n{ref}\nvs\n{got}"
        )


def _worker_one_rank_nonzero(rank, world_size):
    """One rank with signal must keep *every* rank in the step."""
    trainer = _StubTrainer(timestep_batch_size=_CHUNK_BS)
    advantages = torch.tensor([0.0, 0.0]) if rank == 0 else torch.tensor([0.0, 2.0])
    skip = trainer._skip_step_globally({"advantages": advantages})
    assert skip is False, (
        f"rank {rank} decided to skip while another rank had a non-zero advantage; "
        f"that rank would then block on a collective this one never reaches"
    )


def _worker_global_zero_skips(rank, world_size):
    """Signal-free on every rank means every rank skips -- consistently."""
    trainer = _StubTrainer(timestep_batch_size=_CHUNK_BS)
    advantages = torch.zeros(2)
    skip = trainer._skip_step_globally({"advantages": advantages})
    assert skip is True, f"rank {rank} refused to skip a globally signal-free step"

    # Below the 1e-6 threshold is still "zero" and must agree across ranks.
    tiny = torch.full((2,), 1e-9) * (rank + 1)
    assert trainer._skip_step_globally({"advantages": tiny}) is True

    # No advantages at all: no collective is issued, so this must not block. The other
    # rank reaches the same branch, which is why it is safe to assert here.
    assert trainer._skip_step_globally({}) is False


# --------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------


class TestDDPCollectiveStructure(unittest.TestCase):
    """Each case spawns its own pair of gloo ranks; the worker holds the assertions."""

    def test_unequal_chunk_counts_sync_once(self):
        run_distributed(_worker_unequal_chunks)

    def test_gradients_equal_across_ranks(self):
        run_distributed(_worker_grads_equal_across_ranks)

    def test_chunked_matches_unchunked(self):
        run_distributed(_worker_chunked_matches_unchunked)

    def test_one_rank_nonzero_does_not_skip(self):
        run_distributed(_worker_one_rank_nonzero)

    def test_global_zero_skips_consistently(self):
        run_distributed(_worker_global_zero_skips)


class TestComputeLossFailFast(unittest.TestCase):
    """Single-process guards: `compute_loss` must reject the calls it cannot serve."""

    def test_compute_loss_rejects_no_grad(self):
        """The eval/prediction path must fail loudly, not backward under no_grad."""
        trainer = _StubTrainer(timestep_batch_size=_CHUNK_BS)
        model = _TinyPolicy(seed=0)
        inputs = {
            "policy_outputs": [_make_policy_output(2, 6, seed=400)],
            "advantages": torch.tensor([1.0, -1.0]),
        }
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "requires grad to be enabled"):
                trainer.compute_loss(model, inputs)

    def test_compute_loss_rejects_return_outputs(self):
        trainer = _StubTrainer(timestep_batch_size=_CHUNK_BS)
        with self.assertRaisesRegex(ValueError, "does not support returning outputs"):
            trainer.compute_loss(_TinyPolicy(seed=0), {}, return_outputs=True)


if __name__ == "__main__":
    # verbosity=2 names each case as it runs, so a hang is attributable while it hangs.
    unittest.main(verbosity=2)
