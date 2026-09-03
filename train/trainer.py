#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import contextlib
import gc
import inspect
import json
import os
import warnings
from collections import deque
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import torch
import wandb
from accelerate.utils import gather
from accelerate.utils import gather_object
from datasets import Dataset
from datasets import IterableDataset
from torch import nn
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback
from trl.data_utils import is_conversational
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from common.models.policy import HIDDEN_POLICY_TYPES
from common.models.policy import REPLAYABLE_POLICY_TYPES
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import print_prompt_completions_sample

from common.generation.generation import generate_unified
from common.generation.sampling import bernoulli_batch_loglik
from common.generation.sampling import categorical_batch_loglik
from common.generation.sampling import categorical_entropy
from common.generation.sampling import dpls_batch_loglik
from common.s3 import S3UploadCallback

try:
    import rich  # noqa: F401

    _rich_available = True
except ImportError:
    _rich_available = False

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _uses_block_unmask(args) -> bool:
    """remasking='block_unmask_policy': both heads packed side by side per timestep.

    A free function over ``args`` rather than a Trainer attribute, so the stubs that
    borrow Trainer's methods without subclassing it (tests/test_ddp_chunked_backward.py)
    resolve it too.
    """
    return getattr(args, "remasking", None) == "block_unmask_policy"


class Trainer(GRPOTrainer):
    def __init__(
        self,
        model,
        dllm: nn.Module,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (
            None,
            None,
        ),
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=None,
        )
        torch._dynamo.config.capture_scalar_outputs = True
        self.dllm = torch.compile(dllm)
        self.dllm.eval()

        # Initialize buffering for multi-iteration training
        self._buffered_inputs = None
        self._step = 0

        if self.args.remasking in ("policy", "block_policy", "block_unmask_policy"):
            assert self.beta == 0.0, "Beta must be 0.0 for policy-based remasking"

        # Gradient accumulation not supported with current buffering logic
        assert self.args.gradient_accumulation_steps == 1, (
            "gradient_accumulation_steps must be 1 (current buffering does not support gradient accumulation)"
        )

        # The buffering in _prepare_inputs and the advantage process_slice both
        # assume each dataloader batch is exactly per_device_train_batch_size,
        # i.e. generation_batch_size == per_device_train_batch_size * num_processes.
        # TRL silently sets steps_per_generation > 1 otherwise, which misaligns them.
        assert self.args.steps_per_generation == 1, (
            f"steps_per_generation must be 1, got {self.args.steps_per_generation}. "
            f"Set generation_batch_size ({self.args.generation_batch_size}) equal to "
            f"per_device_train_batch_size * num_processes "
            f"({self.args.per_device_train_batch_size} * {self.accelerator.num_processes})."
        )

        # Replay the policy's per-timestep inputs in compute_loss instead of buffering
        # them from the rollout. Worth it exactly when the buffer is the (B,T,L,d_model)
        # one, and mandatory once the dLLM is trainable, since its activations cannot be
        # stored across timesteps at all.
        if self.args.replay_policy_inputs is not None:
            self.replay_policy_inputs = self.args.replay_policy_inputs
        else:
            self.replay_policy_inputs = (
                self.args.train_dllm
                or self.args.policy_type in REPLAYABLE_POLICY_TYPES
            )
        if self.replay_policy_inputs:
            # Refuse rather than silently buffer: the block_policy decode path fills its
            # own rec_c and would ignore the flag, which looks like it worked.
            assert self.args.policy_type in REPLAYABLE_POLICY_TYPES, (
                "replay_policy_inputs is only implemented for the per-step hidden "
                f"policies {REPLAYABLE_POLICY_TYPES}, got "
                f"policy_type={self.args.policy_type!r}. The block_policy path records "
                "one decision per block, so its buffer is ~30x smaller and does not "
                "need replaying."
            )
            # The ES branch builds its own rollout and never records a state history.
            assert not self.args.es_thresholds, (
                "expert steering (es_thresholds) with replay_policy_inputs is not "
                "implemented; the ES rollout records no state history to replay"
            )
        if self.args.train_dllm:
            # The loss path is correct for a trainable backbone, but nothing here puts
            # 8B of parameters into the optimizer or shards their Adam state.
            raise NotImplementedError(
                "train_dllm=True makes compute_loss propagate gradients into the dLLM, "
                "but the optimizer/sharding side is not wired up (8B of Adam state "
                "needs ZeRO/FSDP or LoRA). Remove the flag or implement that first."
            )

        # Track recent rewards for best checkpoint saving
        self.train_reward_queue = deque(
            maxlen=10 * self.args.gradient_accumulation_steps
        )
        self.train_reward_best = -float("inf")
        self.train_reward_best_step = 0
        self.effective_steps = 0
        self.s3_callback = None
        for callback in callbacks:
            if isinstance(callback, S3UploadCallback):
                self.s3_callback = callback
                break

    def _inner_training_loop(self, batch_size=None, *args, **kwargs):
        # On resume, transformers restores _train_batch_size from the
        # checkpoint's trainer_state.json (auto_find_batch_size support). When
        # per_device_train_batch_size changes across resumes (e.g. 8-GPU -> 4-GPU),
        # the restored value shrinks the dataloader batches while the advantage
        # process_slice still uses args, tripping the group-size assert in
        # compute_loss. Always train with the batch size from args.
        if batch_size is not None and batch_size != self.args.train_batch_size:
            print(
                f"Ignoring checkpoint-restored train batch size {batch_size}; "
                f"using {self.args.train_batch_size} from args"
            )
            batch_size = self.args.train_batch_size
            self._train_batch_size = batch_size
        return super()._inner_training_loop(batch_size, *args, **kwargs)

    def train(self, *args, **kwargs):
        """Override train to save final checkpoint at end of training."""
        output = super().train(*args, **kwargs)

        if self.accelerator.is_main_process:
            final_step = self.state.global_step
            checkpoint_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{final_step}"
            )

            print(f"\nSaving final checkpoint at step {final_step}")
            unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
            unwrapped_model.save_pretrained(checkpoint_dir)
            self.state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))

            if self.s3_callback is not None:
                print(
                    f"Uploading checkpoint-{final_step} to s3: {self.args.output_dir}"
                )
                self.s3_callback.on_save(self.args, self.state, self.control)

        return output

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Override training_step to skip optimizer step when advantages are zero."""
        if self._skip_step_globally(inputs):
            # Skip expensive forward/backward passes - no learning signal anywhere
            return torch.tensor(0.0, device=inputs["advantages"].device)

        # Track effective training steps (non-zero advantages)
        self.effective_steps += 1

        # Normal training step for non-zero advantages
        return super().training_step(model, inputs, num_items_in_batch)

    def _skip_step_globally(self, inputs: dict[str, Any]) -> bool:
        """Do *all* ranks agree this step carries no learning signal?

        ``advantages`` is this rank's slice (see ``_generate_and_score_completions``),
        so the local answer differs between ranks whenever one rank's groups happened
        to collapse and another's did not. A rank that returned early while another
        entered backward would leave the others waiting on a collective that never
        comes, so the decision has to be taken jointly: skip only if *every* rank is
        signal-free, and note that the gather below is itself a collective -- every
        rank must reach it, which is why it is unconditional on the value.
        """
        if "advantages" not in inputs:
            # Nothing to agree about, and calling a collective on only the ranks that
            # happen to carry advantages is exactly the desync this guards against.
            return False
        advantages = inputs["advantages"]
        has_signal = (torch.abs(advantages).max() >= 1e-6).to(torch.float32)
        all_signals = self.accelerator.gather(has_signal.reshape(1))
        return bool(all_signals.max().item() < 0.5)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # This loss backwards every chunk but the last itself (see below), so it cannot
        # run under the no-grad that Trainer.prediction_step/evaluate wrap it in: the
        # chunk backwards would raise from inside the loop, or -- worse, with only one
        # chunk -- silently return a graphless tensor. There is no eval loss to report
        # anyway (beta is 0); evaluation goes through eval.eval. Fail loudly instead.
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "compute_loss requires grad to be enabled: it performs its own "
                "per-chunk backward passes and has no no-grad/eval path. Evaluate "
                "checkpoints with `python -m eval.pipeline` instead."
            )
        # Compute the per-token log probabilities for the model

        model.train()

        # Note that the output of the policy
        # is a list containing pointers to tensors of size  (B, T, BL)
        # although if you sum the B dim you get the group size G, these can
        # in general not be stacked because the T
        # can vary between different batches within the group.
        # Therefore, we process it as a list of batches for as long as necessary.
        policy_outputs: list[dict[str, torch.Tensor | tuple[torch.Tensor]]] = inputs[
            "policy_outputs"
        ]

        # ensure the group batches add up to the whole group size (ie sum_i B_i = G)
        group_batch_sizes = [
            policy_output["sampling_masks"].size(0) for policy_output in policy_outputs
        ]
        assert sum(group_batch_sizes) == inputs["advantages"].size(0)

        # Check if ES (Expert Steering) is enabled and compute mixture distribution weights
        has_es = (
            self.args.es_thresholds is not None and len(self.args.es_thresholds) > 0
        )
        if has_es:
            num_es = len(self.args.es_thresholds)
            total_group_size = inputs["advantages"].size(0)
            num_regular = total_group_size - num_es
            device = inputs["advantages"].device
            # Mixture weights: (G/(G+E)) * pi_theta + (1/(G+E)) * dirac
            # Note that we assume a sample can only come from one particular ES dirac
            # (hence why the second factor is not E/(G+E))
            log_weight_theta = torch.log(
                torch.tensor(num_regular / total_group_size, device=device)
            )
            log_weight_dirac = torch.log(
                torch.tensor(1.0 / total_group_size, device=device)
            )

        # Accumulate the loss over the batches in the group
        batch_index_start = 0
        loss_acummulator = 0
        entropy_accumulator = []

        # Every (batch, timestep-chunk) pair is backwarded as soon as it is computed, so
        # no chunk's activations outlive it -- with a 4096-wide policy input those are
        # what dominate memory, and holding all of them until one backward at the end is
        # what made the per-step hidden policy OOM. The loss is a plain sum over
        # timesteps (the GRPO ratio is per-timestep, not per-trajectory), so splitting
        # the backward this way is exactly equivalent.
        #
        # The last chunk is handed back to Trainer.training_step instead, so accelerate
        # still owns the final backward and DDP reduces exactly once. With a single
        # chunk nothing is backwarded here and the path is bit-identical to before.
        group_size = sum(group_batch_sizes)

        # total_chunks decides which chunk is the one synchronising chunk, so it has to
        # be counted exactly the way the loop below chunks -- an off-by-one here means
        # either no reducing backward or two of them, i.e. a hang.
        def _chunk_bs(t):
            bs = self.args.timestep_batch_size if self.args.timestep_batch_size else t
            assert bs > 0, f"{self.args.timestep_batch_size=} must be positive or None"
            return bs

        total_chunks = sum(
            -(-po["sampling_masks"].size(1) // _chunk_bs(po["sampling_masks"].size(1)))
            for po in policy_outputs
        )
        chunks_done = 0
        final_chunk_loss = None
        for batch_idx, batch_policy_output in enumerate(policy_outputs):
            batch_sampling_masks = batch_policy_output[
                "sampling_masks"
            ]  # will need these later

            batch_index_end = batch_index_start + batch_sampling_masks.size(0)

            # Detect if this is an ES batch (last batch in the group)
            is_es_batch = has_es and batch_idx == len(policy_outputs) - 1

            B, T, _ = batch_sampling_masks.shape

            timestep_bs = _chunk_bs(T)
            # Hoisted: the per-chunk loss below needs the same normaliser the whole
            # batch used, and it is a function of the full mask, which is already here.
            num_active_steps = batch_sampling_masks.any(dim=-1).sum(dim=-1)  # (B,)
            assert (num_active_steps > 0).all(), (
                "At least one batch element was active for < 1 steps?"
            )
            for time_step_idx in range(0, T, timestep_bs):
                chunks_done += 1
                is_final_chunk = chunks_done == total_chunks
                # DDP decides whether to prepare for reduction when forward() runs,
                # not when backward() does, so this has to wrap the policy forward
                # below. Wrapping only the backward leaves every chunk syncing.
                #
                # Only the final chunk syncs, and its backward is the one
                # Trainer.training_step performs, so every rank does exactly ONE
                # reducing backward regardless of how many chunks it has. That is
                # load-bearing: T varies per rank with the rollout length, so a
                # per-chunk collective would mismatch across ranks and hang.
                sync_ctx = (
                    contextlib.nullcontext()
                    if is_final_chunk
                    else self.accelerator.no_sync(model)
                )
                with sync_ctx:
                    # Get this batch's data
                    time_step_batch_sampling_masks = batch_sampling_masks[
                        :, time_step_idx : time_step_idx + timestep_bs, :
                    ]
                    time_step_batch_samples = batch_policy_output["samples"][
                        :, time_step_idx : time_step_idx + timestep_bs, :
                    ]

                    ### Prepare time-batched inputs
                    # A None slot is the hidden stream the rollout did not buffer; recompute
                    # exactly this chunk's worth and let it die with the chunk's backward.
                    replayed_hidden = None
                    if batch_policy_output.get("state_history") is not None:
                        replayed_hidden = self._replay_hidden_states(
                            batch_policy_output,
                            time_step_idx,
                            time_step_idx + timestep_bs,
                        )

                    time_step_batch_policy_inputs = []
                    for ptdi in batch_policy_output["policy_inputs"]:
                        if isinstance(ptdi, torch.Tensor):
                            # Find time dim and slice only at that dim
                            time_dim = 1
                            assert ptdi.size(time_dim) == T, (
                                f"ptdi of shape {ptdi.shape=} did not match {T=} "
                                f"at expected {time_dim=}"
                            )
                            slices = [slice(None)] * ptdi.ndim  # by default get everything
                            slices[time_dim] = slice(
                                time_step_idx, time_step_idx + timestep_bs
                            )  # at time_dim get only what belongs to this time-batch

                            time_step_batch_policy_inputs.append(ptdi[tuple(slices)])
                        elif ptdi is None:
                            assert replayed_hidden is not None, (
                                "policy_inputs has an empty slot but the rollout recorded no "
                                "state_history to replay it from"
                            )
                            time_step_batch_policy_inputs.append(replayed_hidden)
                        else:
                            # Non-tensors just get propagated
                            time_step_batch_policy_inputs.append(ptdi)

                    logps_timestep = self._get_per_timestep_logps_block(
                        model=model,
                        samples=time_step_batch_samples,
                        sampling_masks=time_step_batch_sampling_masks,
                        policy_inputs=time_step_batch_policy_inputs,
                        sampling_mode=self.args.sampling_mode,
                        return_entropy=True,
                    )  # (B, timestep_bs), entropy (scalar)

                    if isinstance(logps_timestep, tuple):
                        logps_timestep, entropy = logps_timestep
                        entropy_accumulator.append(entropy.detach())
                    else:
                        # Backward compatibility if return_entropy=False
                        pass

                    # For ES batches, adjust NEW log probabilities with mixture distribution
                    if is_es_batch:
                        logps_timestep = torch.logaddexp(
                            log_weight_theta + logps_timestep, log_weight_dirac
                        )

                    # Get old log probabilities
                    old_logps_slice = batch_policy_output["old_per_timestep_logps"][
                        :, time_step_idx : time_step_idx + timestep_bs
                    ].detach()

                    # For ES batches, adjust OLD log probabilities with mixture distribution
                    if is_es_batch:
                        old_logps_slice = torch.logaddexp(
                            log_weight_theta + old_logps_slice, log_weight_dirac
                        )

                    coeff_1 = torch.exp(
                        logps_timestep - old_logps_slice
                    )  # (B, timestep_bs)
                    coeff_2 = torch.clamp(
                        coeff_1, 1 - self.args.epsilon, 1 + self.args.epsilon
                    )

                    # Get the advantages corresponding to this batch.
                    # Note that the "advantages" are flat of shape (G,),
                    # so we do some indexing to keep track of where the
                    # current batch is.
                    batch_advantages = (
                        inputs["advantages"][batch_index_start:batch_index_end]
                        .detach()
                        .view((-1,) + (1,) * (coeff_1.ndim - 1))
                    )  # (B, 1)

                    per_timestep_loss1 = coeff_1 * batch_advantages
                    per_timestep_loss2 = coeff_2 * batch_advantages
                    per_timestep_loss = torch.min(per_timestep_loss1, per_timestep_loss2)

                    # Only include the loss for the active timesteps
                    per_timestep_loss *= time_step_batch_sampling_masks.any(dim=-1).to(
                        per_timestep_loss.dtype
                    )
                    # Same normalisation the whole-batch reduction used to apply, just
                    # applied per chunk: sum over this chunk's timesteps, divide by the
                    # batch element's active-step count, then by the group size.
                    chunk_loss = (
                        -(per_timestep_loss.sum(dim=-1) / num_active_steps).sum()
                        / group_size
                    )
                    if is_final_chunk:
                        final_chunk_loss = chunk_loss
                    else:
                        self.accelerator.backward(chunk_loss)
                    loss_acummulator = loss_acummulator + chunk_loss.detach()

                    del (
                        chunk_loss,
                        logps_timestep,
                        coeff_1,
                        coeff_2,
                        per_timestep_loss1,
                        per_timestep_loss2,
                        per_timestep_loss,
                    )
                    torch.cuda.empty_cache()

            # Next batch starts where the current one left off
            batch_index_start = batch_index_end

        assert self.beta == 0.0, (
            f"TODO non-zero {self.beta=} not supported at this time"
        )

        # Already normalised per chunk. The returned tensor carries only the final
        # chunk's graph -- every earlier chunk was backwarded above -- while the added
        # constant makes its value the whole loss, which is what gets logged.
        assert chunks_done == total_chunks, (
            f"chunked backward ran {chunks_done} chunks but planned {total_chunks}; the "
            "final (synchronising) chunk would be the wrong one, which deadlocks DDP"
        )
        assert final_chunk_loss is not None, "compute_loss produced no chunks"
        loss = final_chunk_loss + (loss_acummulator - final_chunk_loss.detach())

        # Log entropy if collected
        if entropy_accumulator:
            mean_entropy = torch.stack(entropy_accumulator).mean(dim=0)
            if mean_entropy.ndim == 0:
                self._metrics["train"]["entropy"].append(
                    self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                )
            else:
                # block_unmask_policy: [per-position Bernoulli entropy, block-size
                # categorical entropy]. Logged separately -- the block one is the
                # collapse signal (formulation.md 9), the other the decoding-rate one.
                gathered = self.accelerator.gather_for_metrics(
                    mean_entropy.unsqueeze(0)
                ).mean(dim=0)
                self._metrics["train"]["entropy"].append(gathered[0].item())
                self._metrics["train"]["block_entropy"].append(gathered[1].item())

        return loss

    def _replay_hidden_states(self, batch_policy_output, t_start, t_end):
        """Recompute the dLLM hidden states the policy saw at timesteps [t_start, t_end).

        The rollout stored the (B,T,L) token state instead of the (B,T,L,d_model) hidden
        states -- about 2000x smaller -- and the dLLM is deterministic given its input,
        so running it over that state reproduces them exactly. One forward per timestep,
        B sequences at a time: batching the whole chunk into B*t rows would materialise a
        vocab-sized logit tensor t times over, which is what the buffer was avoiding.

        :return: (B, t_end - t_start, L, d_model)
        """
        state = batch_policy_output["state_history"][:, t_start:t_end]  # (B, t, L)
        prompt_ids = batch_policy_output["prompt_ids"]  # (B, P)
        prompt_mask = batch_policy_output["prompt_mask"]  # (B, P)
        P = prompt_ids.size(1)

        # Same mask generate_unified builds: real prompt mask, then ones over the
        # generation region.
        attn = torch.ones(
            (state.size(0), P + state.size(-1)),
            dtype=torch.float,
            device=state.device,
        )
        attn[:, :P] = prompt_mask.float()
        attn = attn.to(self.dllm.dtype)

        hidden_kwargs = {}
        if "final_hidden_state_only" in inspect.signature(self.dllm.forward).parameters:
            hidden_kwargs["final_hidden_state_only"] = True

        # Frozen backbone => no graph to keep. Flipping train_dllm makes the same
        # forward differentiable, which is the whole point of replaying rather than
        # buffering: backbone activations could never have been stored across T.
        ctx = contextlib.nullcontext() if self.args.train_dllm else torch.no_grad()
        out = []
        with ctx:
            for k in range(state.size(1)):
                x = torch.cat([prompt_ids, state[:, k]], dim=1)
                model_output = self.dllm(
                    x,
                    attention_mask=attn,
                    output_hidden_states=True,
                    **hidden_kwargs,
                )
                out.append(model_output.hidden_states[-1][:, P:, :].contiguous())
        return torch.stack(out, dim=1)

    def _get_per_timestep_logps_block(
        self,
        model,
        samples,
        sampling_masks,
        policy_inputs,
        sampling_mode="bernoulli",
        return_entropy=False,
    ):
        """Compute log-probabilities for sampled actions under the policy.

        :param model: policy model
        :param samples: sampled actions
        :param sampling_masks: mask indicating valid positions
        :param policy_inputs: inputs to policy model
        :param sampling_mode: sampling mode ("bernoulli", "dpls")
        :param return_entropy: whether to compute and return entropy
        :return: (log_probs, entropy) or just log_probs if return_entropy=False
        """
        with torch.amp.autocast("cuda", enabled=self.args.fp16):
            logits = model(*policy_inputs)  # (B, T, BL)

        if sampling_mode == "categorical" or _uses_block_unmask(self.args):
            # The two-headed policies return their heads separately; concatenate them
            # into the same layout the rollout recorded so one code path handles both.
            logits = torch.cat(logits, dim=-1)  # (B, T, K_block + K_thres) or (B, T, L + K)

        # Calculate corresponding log-likelihoods under the model
        if _uses_block_unmask(self.args):
            lls = self._block_unmask_joint_loglik(
                samples=samples,
                logits=logits,
                sampling_masks=sampling_masks,
            )
        elif sampling_mode == "categorical":
            lls = self._categorical_joint_loglik(
                samples=samples,
                logits=logits,
                sampling_masks=sampling_masks,
            )
        elif sampling_mode == "dpls":
            lls = dpls_batch_loglik(
                samples=samples,
                utilities=logits,
                stop_logit=self.args.dpls_stop_logit,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif sampling_mode in ["bernoulli", "bernoulli-argmax"]:
            lls = bernoulli_batch_loglik(
                samples,
                logits,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        else:
            raise ValueError(f"Unexpected {sampling_mode=}")

        # Compute entropy if requested
        entropy = None
        if return_entropy:
            if _uses_block_unmask(self.args):
                n = sampling_masks.shape[-1] - 1
                pos_logits, block_logits = logits[..., :n], logits[..., n:]
                pos_mask = sampling_masks[..., :n].float()
                p = torch.sigmoid(pos_logits).clamp(1e-8, 1 - 1e-8)
                pos_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                pos_entropy = (pos_entropy * pos_mask).sum() / pos_mask.sum().clamp(
                    min=1
                )
                dec_mask = sampling_masks[..., n].float()
                block_entropy = categorical_entropy(
                    block_logits, feasible_mask=torch.isfinite(block_logits)
                )
                block_entropy = (block_entropy * dec_mask).sum() / dec_mask.sum().clamp(
                    min=1
                )
                entropy = torch.stack([pos_entropy, block_entropy])
            elif sampling_mode == "categorical":
                # Joint entropy of two independent categoricals is the sum. Averaged
                # over real decisions only -- this is the primary collapse signal
                # (formulation.md 9), so padded slots must not dilute it.
                block_logits, thres_logits = self._split_joint_logits(logits)
                joint = categorical_entropy(
                    block_logits, feasible_mask=torch.isfinite(block_logits)
                ) + categorical_entropy(thres_logits)
                active_mask = sampling_masks[..., 0].float()
                entropy = (joint * active_mask).sum() / active_mask.sum().clamp(min=1)
            elif sampling_mode in ["bernoulli", "bernoulli-argmax"]:
                # For Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
                probs_clamped = torch.sigmoid(logits).clamp(1e-8, 1 - 1e-8)
                entropy = -(
                    probs_clamped * torch.log(probs_clamped)
                    + (1 - probs_clamped) * torch.log(1 - probs_clamped)
                )  # (B, timestep_bs, BL)
                # Average over active positions only
                active_mask = sampling_masks.float()
                entropy = (entropy * active_mask).sum() / active_mask.sum()
            elif sampling_mode == "dpls":
                masked_logits = logits.masked_fill(~sampling_masks, float("-inf"))
                probs = torch.softmax(masked_logits, dim=-1)
                probs_clamped = probs.clamp(1e-8, 1.0)
                entropy = -(probs * torch.log(probs_clamped)).sum(dim=-1)
                active_mask = sampling_masks.any(dim=-1).float()
                entropy = (entropy * active_mask).sum() / active_mask.sum()

        del logits
        torch.cuda.empty_cache()

        if return_entropy:
            return lls, entropy
        return lls

    def _split_joint_logits(self, logits: torch.Tensor):
        """Split the concatenated (block size, threshold) head logits.

        Both heads share one tensor so that the generic time-axis slicing in
        compute_loss works unchanged; they are split back here.

        :param logits: (..., K_block + K_thres)
        :return: ((..., K_block), (..., K_thres))
        """
        n_block = len(self.args.block_size_candidates)
        return logits[..., :n_block], logits[..., n_block:]

    def _categorical_joint_loglik(
        self,
        samples: torch.Tensor,
        logits: torch.Tensor,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Joint log-prob of one (block size, threshold) action pair.

        The heads are conditionally independent given the state, so the joint log-prob
        is the sum of the two (formulation.md 5.3).

        :param samples: (..., 2) action indices [block idx, threshold idx]
        :param logits: (..., K_block + K_thres) concatenated head logits
        :param sampling_masks: (..., 1) True on real decisions, False on padding
        :return: (...,) joint log-likelihood
        """
        block_logits, thres_logits = self._split_joint_logits(logits)
        active = sampling_masks[..., 0]
        return categorical_batch_loglik(
            samples[..., 0],
            block_logits,
            action_mask=active,
            dtype=self.args.loglikelihood_dtype,
        ) + categorical_batch_loglik(
            samples[..., 1],
            thres_logits,
            action_mask=active,
            dtype=self.args.loglikelihood_dtype,
        )

    def _block_unmask_joint_loglik(
        self,
        samples: torch.Tensor,
        logits: torch.Tensor,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Joint log-prob of one timestep of remasking='block_unmask_policy'.

        The per-position Bernoulli term is always present on active steps; the
        block-size categorical term only on the step that entered a new block (the
        last column of sampling_masks). Both heads are conditionally independent
        given the state, so the joint log-prob is the sum. The rollout's old
        log-probs and this recomputation must use exactly this decomposition, or the
        GRPO ratio is not 1 on the first inner iteration.

        :param samples: (..., L+1) long, [bernoulli draws | block-size index]
        :param logits: (..., L+K) [unmask logits | block-size logits]
        :param sampling_masks: (..., L+1) bool, [in-block masked positions | decided]
        :return: (...,) joint log-likelihood
        """
        n = sampling_masks.shape[-1] - 1
        pos_ll = bernoulli_batch_loglik(
            samples[..., :n],
            logits[..., :n],
            mask_index=sampling_masks[..., :n],
            dtype=self.args.loglikelihood_dtype,
        )
        block_ll = categorical_batch_loglik(
            samples[..., n],
            logits[..., n:],
            action_mask=sampling_masks[..., n],
            dtype=self.args.loglikelihood_dtype,
        )
        return pos_ll + block_ll

    def _compute_mask_loglikelihood(
        self,
        samples: torch.Tensor,
        sampling_inputs: torch.Tensor,
        sampling_masks: torch.Tensor,
    ) -> torch.Tensor:
        if _uses_block_unmask(self.args):
            return self._block_unmask_joint_loglik(
                samples=samples,
                logits=sampling_inputs,
                sampling_masks=sampling_masks,
            )
        elif self.args.sampling_mode == "categorical":
            return self._categorical_joint_loglik(
                samples=samples,
                logits=sampling_inputs,
                sampling_masks=sampling_masks,
            )
        elif self.args.sampling_mode in ["bernoulli", "bernoulli-argmax"]:
            return bernoulli_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        elif self.args.sampling_mode == "dpls":
            return dpls_batch_loglik(
                samples=samples,
                utilities=sampling_inputs,
                stop_logit=self.args.dpls_stop_logit,
                mask_index=sampling_masks,
                dtype=self.args.loglikelihood_dtype,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {self.args.sampling_mode}")

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                # Generate new completions
                inputs = self._generate_and_score_completions(inputs)
                # Store for reuse in next num_iterations-1 steps
                self._buffered_inputs = inputs
            else:
                # Reuse buffered completions
                inputs = self._buffered_inputs
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        # Remove assertion - we now support mixed dataset types in a batch
        # assert len(set([x["dataset_type"] for x in inputs])) == 1

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # Need to add the gen_prefix to the prompt for KodCode (per-sample check)
        for i, example in enumerate(inputs):
            if example["dataset_type"] == "kodcode":
                prompts_text[i] = prompts_text[i] + example["gen_prefix"]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = HFTrainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        temperature = self.args.temperature or 0.0

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            num_steps_all = []
            force_es_thresholds = None
            if self.args.es_thresholds:
                # TODO: For now we are hardcoding BL=32 for ES samples.
                force_es_thresholds = torch.tensor(
                    self.args.es_thresholds,
                    dtype=unwrapped_model.dtype,
                    device=unwrapped_model.device,
                ).unsqueeze(-1)
            if self.args.remasking in ("policy", "block_policy", "block_unmask_policy"):
                policy_outputs_all = []
                still_masked_all = []
                block_sizes_all = []
                thresholds_all = []
                block_decisions_all = []
                is_block_policy = self.args.remasking == "block_policy"
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx]
                    batch_prompt_mask = prompt_mask[i:end_idx]

                    extra_kwargs = {}
                    if is_block_policy:
                        # block_length is unused here: the policy picks it per block,
                        # and the threshold likewise comes from the policy, not
                        # self.args.thres.
                        extra_kwargs = {
                            "block_size_candidates": tuple(
                                self.args.block_size_candidates
                            ),
                            "threshold_candidates": tuple(
                                self.args.threshold_candidates
                            ),
                        }
                    elif _uses_block_unmask(self.args):
                        # Block size from the policy; within the block the unmask
                        # head decides, so there is no threshold at all.
                        extra_kwargs = {
                            "block_size_candidates": tuple(
                                self.args.block_size_candidates
                            ),
                            "block_sampling_mode": self.args.block_sampling_mode,
                        }

                    result = generate_unified(
                        model=self.dllm,
                        prompt=batch_prompt_ids,
                        remasking=self.args.remasking,
                        policy=unwrapped_model,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        mask_id=self.args.mask_id,
                        sampling_mode=self.args.sampling_mode,
                        dpls_stop_logit=self.args.dpls_stop_logit,
                        full_context=self.args.policy_full_context,
                        confidences_top_p=self.args.confidences_top_p,
                        model_type=self.args.model_type,
                        attention_mask=batch_prompt_mask,
                        replay_policy_inputs=self.replay_policy_inputs,
                        **extra_kwargs,
                    )
                    if is_block_policy or _uses_block_unmask(self.args):
                        block_sizes_all.append(result.block_sizes_chosen)
                        thresholds_all.append(result.thresholds_chosen)
                        block_decisions_all.append(result.block_decisions)

                    # Extract values from NamedTuple
                    batch_prompt_completion_ids = result.sequences
                    batch_sampling_inputs = result.sampling_inputs
                    batch_samples = result.samples
                    batch_sampling_masks = result.sampling_masks
                    num_steps = result.steps_taken
                    batch_policy_inputs = result.policy_inputs
                    still_masked = result.still_masked

                    # Compute log-likelihood based on sampling mode
                    mask_ll = self._compute_mask_loglikelihood(
                        samples=batch_samples,
                        sampling_inputs=batch_sampling_inputs,
                        sampling_masks=batch_sampling_masks,
                    )

                    policy_outputs_all.append(
                        {
                            "samples": batch_samples,
                            "sampling_masks": batch_sampling_masks,
                            "old_per_timestep_logps": mask_ll,
                            "prompt_length": batch_prompt_ids.shape[1],
                            "sampling_inputs": batch_sampling_inputs,
                            "policy_inputs": batch_policy_inputs,
                            # Replay keys: the (B,T,L) token state the policy saw, plus
                            # what it takes to rebuild the full sequence and mask.
                            "state_history": result.state_history,
                            "prompt_ids": batch_prompt_ids,
                            "prompt_mask": batch_prompt_mask,
                        }
                    )
                    num_steps_all.append(num_steps)
                    still_masked_all.append(still_masked)
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)
                    # Removed gc.collect() and empty_cache() from inner loop for better GPU utilization

                if force_es_thresholds is not None:
                    es_prompt_ids = (
                        prompt_ids[0:1]
                        .expand(
                            force_es_thresholds.size(0),
                            *[-1] * (prompt_ids.ndim - 1),
                        )
                        .contiguous()
                    )
                    es_prompt_mask = (
                        prompt_mask[0:1]
                        .expand(
                            force_es_thresholds.size(0),
                            *[-1] * (prompt_mask.ndim - 1),
                        )
                        .contiguous()
                    )
                    result = generate_unified(
                        model=self.dllm,
                        prompt=es_prompt_ids,
                        remasking="fastdllm",
                        thres=force_es_thresholds,
                        policy=unwrapped_model,
                        gen_length=gen_length,
                        block_length=32,  # TODO: in the future we may not want to hardcode this
                        temperature=temperature,
                        mask_id=self.args.mask_id,
                        full_context=self.args.policy_full_context,
                        confidences_top_p=self.args.confidences_top_p,
                        temperature_policy=1.0,
                        model_type=self.args.model_type,
                        attention_mask=es_prompt_mask,
                    )

                    # Extract values from NamedTuple
                    es_prompt_completion_ids = result.sequences
                    es_sampling_inputs = result.sampling_inputs
                    es_samples = result.samples
                    es_sampling_masks = result.sampling_masks
                    es_num_steps = result.steps_taken
                    es_policy_inputs = result.policy_inputs
                    # still_masked ignored for ES

                    # Compute log-likelihood based on sampling mode
                    es_mask_ll = self._compute_mask_loglikelihood(
                        samples=es_samples,
                        sampling_inputs=es_sampling_inputs,
                        sampling_masks=es_sampling_masks,
                    )

                    policy_outputs_all.append(
                        {
                            "samples": es_samples,
                            "sampling_masks": es_sampling_masks,
                            "old_per_timestep_logps": es_mask_ll,
                            "prompt_length": es_prompt_ids.shape[1],
                            "sampling_inputs": es_sampling_inputs,
                            "policy_inputs": es_policy_inputs,
                        }
                    )
                    num_steps_all.append(es_num_steps)
                    prompt_completion_ids_all.append(es_prompt_completion_ids)
                    # Removed gc.collect() and empty_cache() from inner loop for better GPU utilization

                prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
                num_steps = torch.cat(num_steps_all, dim=0)
                still_masked = torch.cat(still_masked_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        if self.args.es_thresholds:
            # Add copies for ES, if needed
            # TODO: assuming we can reuse inputs[0],
            # should be safe since all inputs should be the same on the gpu anyway
            num_es_samples = len(self.args.es_thresholds)
            inputs.extend([inputs[0]] * num_es_samples)
            prompts.extend([inputs[0]["prompt"]] * num_es_samples)
        assert len(completion_ids) == len(prompts) == len(inputs), (
            f"{len(completion_ids)=} ?!= {len(prompts)=} ?!= {len(inputs)=}"
        )

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

            if reward_func_name in [
                "lm_eval_flex_mult_reward",
                "lm_eval_flex_add_reward",
                "xml_mult_reward",
                "xml_add_reward",
                "math_correctness_mult_reward",
                "mixed_correctness_mult_reward_func",
                "mixed_correctness_add_reward_func",
                "mixed_correctness_reward_func",
                "kodcode_correctness_mult_reward",
            ]:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    n_steps=num_steps,
                    L=gen_length,
                    alpha=self.args.alpha_compute_reward,
                    step=self._step,
                    run_name=self.args.output_dir,
                    pos_reward=self.args.alpha_correctness_reward,
                    **reward_kwargs,
                )
            elif reward_func_name == "lm_eval_harness_flexible_match_reward_func":
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    pos_reward=self.args.alpha_correctness_reward,
                    **reward_kwargs,
                )
            else:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )

            assert len(output_reward_func) == len(prompts) == len(completion_ids), (
                f"{len(output_reward_func)=} != {len(prompts)=} = {len(completion_ids)=}"
            )
            # Convert None values to NaN
            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]
            assert len(output_reward_func) == len(prompts), (
                f"{len(output_reward_func)=} != {len(prompts)=}"
            )

            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        group_size = self.num_generations + len(self.args.es_thresholds or [])
        mean_grouped_rewards = rewards.view(-1, group_size).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, group_size).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(group_size, dim=0)
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation (policy only for metrics)
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        # Slice out this process's advantages
        # Each process has per_device_train_batch_size items (potentially multiple groups)
        items_per_process = self.args.per_device_train_batch_size + (
            len(self.args.es_thresholds) if self.args.es_thresholds else 0
        )
        process_slice = slice(
            self.accelerator.process_index * items_per_process,
            (self.accelerator.process_index + 1) * items_per_process,
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float()

        # For rewards and other metrics (eg completion length) inferred from the prompt_completion_ids,
        # we need to slice out ES samples (if any) so as to not pollute the logs
        num_es = len(self.args.es_thresholds) if self.args.es_thresholds else 0
        if num_es > 0:
            num_processes = self.accelerator.num_processes
            policy_samples_per_process = group_size - num_es
            post_gathering_policy_only_index = torch.ones(
                completion_length.shape[0],
                dtype=torch.bool,
                device=completion_length.device,
            )
            for proc_idx in range(num_processes):
                start_idx = proc_idx * group_size
                es_start = start_idx + policy_samples_per_process
                es_end = start_idx + group_size
                post_gathering_policy_only_index[es_start:es_end] = False
        else:
            post_gathering_policy_only_index = slice(None)

        completion_length = (
            completion_length[post_gathering_policy_only_index].mean().item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)
        self._metrics[mode]["effective_steps"].append(self.effective_steps)

        still_masked = gather(still_masked)  # (N_GPUs * G,)
        still_masked = (still_masked.float() / gen_length).mean()
        self._metrics[mode]["still_masked"].append(still_masked.item())

        # Metrics: Calculate mean reward per function, but only for samples where the function was applied
        # and the sample actually came from the policy (not ES)
        rewards_per_func_policy = rewards_per_func[post_gathering_policy_only_index]
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func_policy[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        rewards_policy = rewards[post_gathering_policy_only_index]
        self._metrics[mode]["reward"].append(rewards_policy.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log ES advantages if present
        if num_es > 0:
            # Gather all advantages across processes
            advantages_gathered = gather(advantages)
            # Extract ES advantages (inverse of policy-only index)
            advantages_es = advantages_gathered[~post_gathering_policy_only_index]
            self._metrics[mode]["es_advantage_mean"].append(advantages_es.mean().item())

        if (
            self.args.save_best_checkpoint
            and mode == "train"
            and self.accelerator.is_main_process
            and self.state.global_step % self.num_iterations == 0
        ):
            self.train_reward_queue.append(rewards_policy.mean().item())
            if np.mean(self.train_reward_queue) > self.train_reward_best:
                self.train_reward_best = np.mean(self.train_reward_queue)
                self.train_reward_best_step = self.state.global_step

                _output_dir = os.path.join(self.args.output_dir, "checkpoint-best")
                unwrapped_model = self.accelerator.unwrap_model(self.model_wrapped)
                unwrapped_model.save_pretrained(_output_dir)
                self.state.save_to_json(os.path.join(_output_dir, "trainer_state.json"))
                with open(
                    os.path.join(_output_dir, "best_train_reward.json"), "w"
                ) as f:
                    json.dump(
                        {
                            "best_train_reward": self.train_reward_best,
                            "best_train_reward_step": self.train_reward_best_step,
                        },
                        f,
                    )

                print(
                    f"Saved checkpoint-best at step {self.train_reward_best_step} with train reward {self.train_reward_best}"
                )
                if self.s3_callback is not None:
                    # use the callback to push the checkpoint to s3
                    print(f"Uploading checkpoint-best to s3: {self.args.output_dir}")
                    self.s3_callback.on_save(
                        self.args, self.state, self.control, best=True
                    )

        # Log metrics to detect the collapse to 0 policy. The block-size histogram and
        # the per-position unmask-probability diagnostics are independent blocks:
        # block_policy gets only the first, policy only the second, and
        # block_unmask_policy both.
        if self.args.remasking in ("block_policy", "block_unmask_policy"):
            # Per formulation.md 9 the action histogram and the entropy are the
            # collapse signal that decides whether to abandon the run early.
            device = num_steps.device
            cand_b = torch.tensor(
                self.args.block_size_candidates, device=device, dtype=torch.long
            )
            # block_unmask_policy has no threshold axis at all.
            has_thres = thresholds_all[0] is not None
            cand_t = torch.tensor(
                self.args.threshold_candidates if has_thres else [],
                device=device,
                dtype=torch.float32,
            )

            # Accumulate over generation chunks rather than concatenating them:
            # each chunk packs its decisions to its own T, so a cat along dim 0
            # would hit a size mismatch on dim 1 whenever there is >1 chunk.
            counts_b = torch.zeros(len(cand_b), device=device)
            counts_t = torch.zeros(len(cand_t), device=device)
            totals = torch.zeros(3, device=device)  # [n_decisions, sum_b, sum_t]
            per_row_blocks = []
            for bs_chunk, th_chunk, act in zip(
                block_sizes_all, thresholds_all, block_decisions_all
            ):
                # act: (B, T) True on the slots that carry a real decision
                counts_b += (
                    ((bs_chunk.unsqueeze(-1) == cand_b) & act.unsqueeze(-1))
                    .sum(dim=(0, 1))
                    .float()
                )
                sum_t = torch.zeros((), device=device)
                if has_thres:
                    counts_t += (
                        (
                            ((th_chunk.unsqueeze(-1) - cand_t).abs() < 1e-6)
                            & act.unsqueeze(-1)
                        )
                        .sum(dim=(0, 1))
                        .float()
                    )
                    sum_t = (th_chunk * act).sum().float()
                totals += torch.stack(
                    [
                        act.sum().float(),
                        (bs_chunk * act).sum().float(),
                        sum_t,
                    ]
                )
                per_row_blocks.append(act.sum(dim=-1).float())

            # gather_for_metrics uses all_gather, which requires identical shapes on
            # every rank. The number of decisions differs per rank (rollouts pick
            # different block counts), so reduce to fixed-shape summaries first and
            # only then gather. Gathering the ragged per-decision tensor directly
            # would hang under NCCL.
            counts_b = self.accelerator.gather_for_metrics(counts_b.unsqueeze(0)).sum(0)
            counts_t = self.accelerator.gather_for_metrics(counts_t.unsqueeze(0)).sum(0)
            totals = self.accelerator.gather_for_metrics(totals.unsqueeze(0)).sum(0)
            # (B,) is the same on every rank, so this one gathers safely as-is.
            per_row_blocks = self.accelerator.gather_for_metrics(
                torch.cat(per_row_blocks, dim=0)
            )

            n_dec = totals[0].clamp(min=1.0)
            self._metrics[mode]["block_size/mean"].append((totals[1] / n_dec).item())
            self._metrics[mode]["num_blocks_mean"].append(
                per_row_blocks.mean().item()
            )
            for i, b in enumerate(self.args.block_size_candidates):
                self._metrics[mode][f"block_size/frac_{b}"].append(
                    (counts_b[i] / n_dec).item()
                )
            if has_thres:
                self._metrics[mode]["threshold/mean"].append(
                    (totals[2] / n_dec).item()
                )
                for i, t in enumerate(self.args.threshold_candidates):
                    self._metrics[mode][f"threshold/frac_{t}"].append(
                        (counts_t[i] / n_dec).item()
                    )
        if self.args.remasking in ("policy", "block_unmask_policy"):
            avg_us_all = []
            max_us_all = []
            non_zero_active_us_all = []
            non_zero_bs_timesteps_all = []
            for i in range(
                # Drop last batch, corresponding to ES samples, if present
                len(policy_outputs_all) - (1 if self.args.es_thresholds else 0)
            ):
                sampling_inputs = policy_outputs_all[i][
                    "sampling_inputs"
                ]  # Contains logits for both Bernoulli and DPLS
                samples = policy_outputs_all[i][
                    "samples"
                ]  # One-hot bernoulli outcomes for Bernoulli, ordered indices for PL
                ms = policy_outputs_all[i]["sampling_masks"]
                if _uses_block_unmask(self.args):
                    # Drop the block-size columns packed after the L positions.
                    n = ms.shape[-1] - 1
                    sampling_inputs = sampling_inputs[..., :n]
                    samples = samples[..., :n].bool()
                    ms = ms[..., :n]

                # Convert to probabilities for consistent logging across sampling modes
                if self.args.sampling_mode == "dpls":
                    # For DPLS, sampling_inputs contains logits, convert to probabilities
                    # Do not normalize over unmasked tokens
                    sampling_inputs = torch.where(
                        ms.any(dim=-1).unsqueeze(-1),
                        sampling_inputs.masked_fill(~ms, float("-inf")),
                        torch.zeros_like(sampling_inputs),
                    )
                    us = torch.softmax(sampling_inputs, dim=-1, dtype=torch.float32)
                    # Similarly samples need to be converted to one-hot
                    # (do not care about their ordering for logging)
                    # Since samples vector may contain padding (-1), we clamp to valid indices
                    # but then dynamically set the value to True/False - making the scatter
                    # a no-op at the padded indices
                    bs = (
                        torch.zeros_like(us, dtype=torch.int)
                        .scatter_add(-1, samples.clamp(min=0), (samples >= 0).int())
                        .bool()
                    )
                else:
                    # For Bernoulli, sampling_inputs contains logits, convert to probabilities with sigmoid
                    us = torch.sigmoid(sampling_inputs)
                    # And samples contains the sampled unmasking indices (one-hot)
                    bs = samples

                # For average unmask probability as well as for proportion of non-zero unmask probability,
                # we average both over time and the block dimension
                avg_us = (us * ms).sum(dim=(-1, -2)) / ms.sum(dim=(-1, -2))
                eps = 0.001
                non_zero_active_us = ((us * ms) > eps).sum(dim=(-1, -2)) / ms.sum(
                    dim=(-1, -2)
                )
                avg_us_all.append(avg_us)
                non_zero_active_us_all.append(non_zero_active_us)

                active_timesteps = ms.any(dim=-1)  # (B, T)
                # For max unmask probability, we aggregate over the block dimension only
                # and then avg over the active timesteps
                max_us = torch.amax(us * ms, dim=(-1))
                max_us = (max_us * active_timesteps).sum(dim=-1) / active_timesteps.sum(
                    dim=-1
                )
                max_us_all.append(max_us)
                # For non-zero bs, we take any over the block dimension and
                # then avg over the active timesteps
                non_zero_bs_timesteps = bs.any(dim=-1)  # (B, T)
                non_zero_bs_timesteps = (non_zero_bs_timesteps * active_timesteps).sum(
                    dim=-1
                ) / active_timesteps.sum(dim=-1)
                non_zero_bs_timesteps_all.append(non_zero_bs_timesteps)

            avg_us_all = torch.cat(avg_us_all, dim=0)
            max_us_all = torch.cat(max_us_all, dim=0)
            non_zero_active_us_all = torch.cat(non_zero_active_us_all, dim=0)
            non_zero_bs_timesteps_all = torch.cat(non_zero_bs_timesteps_all, dim=0)

            avg_us_all = self.accelerator.gather_for_metrics(avg_us_all)
            non_zero_active_us_all = self.accelerator.gather_for_metrics(
                non_zero_active_us_all
            )
            max_us_all = self.accelerator.gather_for_metrics(max_us_all)

            non_zero_bs_timesteps_all = self.accelerator.gather_for_metrics(
                non_zero_bs_timesteps_all
            )

            self._metrics[mode]["mean_unmask_prob"].append(avg_us_all.mean().item())
            self._metrics[mode]["non_zero_unmask_prob"].append(
                non_zero_active_us_all.mean().item()
            )
            self._metrics[mode]["max_unmask_prob"].append(max_us_all.mean().item())

            self._metrics[mode]["non_zero_bs_timesteps_mean"].append(
                non_zero_bs_timesteps_all.mean().item()
            )
            self._metrics[mode]["non_zero_bs_timesteps_std"].append(
                non_zero_bs_timesteps_all.std().item()
            )
            self._metrics[mode]["non_zero_bs_timesteps_min"].append(
                non_zero_bs_timesteps_all.min().item()
            )
            self._metrics[mode]["non_zero_bs_timesteps_max"].append(
                non_zero_bs_timesteps_all.max().item()
            )

        # NFE metrics apply to both action types
        num_steps = self.accelerator.gather_for_metrics(num_steps).float()
        num_steps = num_steps[post_gathering_policy_only_index]
        self._metrics[mode]["num_steps_mean"].append(num_steps.mean().item())
        self._metrics[mode]["num_steps_std"].append(num_steps.std().item())
        self._metrics[mode]["num_steps_min"].append(num_steps.min().item())
        self._metrics[mode]["num_steps_max"].append(num_steps.max().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if _rich_available:
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # clear cuda memory
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "policy_outputs": policy_outputs_all,
        }
