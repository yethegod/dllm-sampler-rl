#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
from trl.trainer.grpo_config import GRPOConfig


@dataclass
class Config(GRPOConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.loglikelihood_dtype is not None:
            assert isinstance(self.loglikelihood_dtype, str)
            if self.loglikelihood_dtype.lower() in {"none", "null"}:
                # should be handled by super already but let's manually catch it just in case it slipped by
                self.loglikelihood_dtype = None
            else:
                try:
                    self.loglikelihood_dtype = getattr(torch, self.loglikelihood_dtype)
                    if not self.loglikelihood_dtype.is_floating_point:
                        raise AttributeError()
                except AttributeError:
                    raise TypeError(
                        f"loglikelihood_dtype = {self.loglikelihood_dtype} is not a valid floating-point torch dtype"
                    )

    def to_dict(self):
        # __post_init__ turns loglikelihood_dtype into a torch.dtype, which is
        # not JSON-serializable. Stringify it so to_json_string() (used by the
        # TensorBoard callback on_train_begin) doesn't crash.
        d = super().to_dict()
        if isinstance(d.get("loglikelihood_dtype"), torch.dtype):
            d["loglikelihood_dtype"] = str(d["loglikelihood_dtype"])
        return d

    # Parameters that control the data preprocessing
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to the base diffusion model."},
    )

    # Diffusion-specific parameters

    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size for generation. If `None`, defaults to effective training batch size."
        },
    )

    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "diffusion block length"},
    )
    temperature: float = field(
        default=0.0,
        metadata={
            "help": "Temperature for Gumbel noise during generation. 0.0 means no noise."
        },
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )
    reward_functions: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of reward function names to use. If None, uses dataset-specific defaults. "
            "See reward_func.py module for available functions."
        },
    )

    policy_type: str = field(
        default="dit_confidence",
        metadata={
            "help": "Type of policy to use. Options: ['dit_hidden', 'dit_hidden_proj', "
            "'dit_confidence', 'dit_block_size', 'dit_block_size_hidden_proj']."
        },
    )

    block_size_candidates: Optional[list[int]] = field(
        default_factory=lambda: [8, 16, 32, 64, 128],
        metadata={
            "help": "Ascending candidate block sizes for the joint (block size, threshold) "
            "policy. Only used by the 'dit_block_size*' policy types."
        },
    )

    threshold_candidates: Optional[list[float]] = field(
        default_factory=lambda: [0.5, 0.7, 0.9],
        metadata={
            "help": "Candidate Fast-dLLM confidence thresholds for the joint policy. "
            "Only used by the 'dit_block_size*' policy types."
        },
    )

    block_size_prior_logits: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Initial per-candidate bias for the block-size head. None means zeros "
            "(near-uniform at init). Must match len(block_size_candidates)."
        },
    )

    policy_num_heads: int = field(
        default=2,
        metadata={"help": "Number of attention heads for policy transformer blocks."},
    )
    policy_hidden_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Hidden dimension for policy transformer blocks. If None, uses model's hidden size."
        },
    )
    policy_feedforward_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Feedforward dimension for policy transformer blocks. If None, uses 4 * hidden_dim."
        },
    )
    policy_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for policy transformer blocks."},
    )
    policy_time_embed_dim: int = field(
        default=256, metadata={"help": "Time embedding dimension for policy adaLN."}
    )
    policy_time_period: float = field(
        default=1,
        metadata={"help": "Max period for sinusoidal time embedding in policy."},
    )
    policy_num_blocks: int = field(
        default=1,
        metadata={"help": "Number of transformer blocks for policy architectures."},
    )

    policy_smart_init: float | None = field(
        default=None,
        metadata={
            "help": "Target mean for smart initialization of policy. "
            "Sets ada ln to identity and output_proj bias to this value, centering initial logits "
            "at the specified target. For DPLS sampling, use 0.0 to match stop_logit "
            "for balanced sampling, or negative values to bias toward stopping earlier. "
            "If None, uses default PyTorch initialization."
        },
    )

    confidences_top_p: int = field(
        default=1,
        metadata={
            "help": "The number of top confidences to use as input to the policy. Only used for dit_confidence policy type."
        },
    )

    alpha_compute_reward: float = field(
        default=0.0,
        metadata={"help": "Weight of the compute term in the reward function."},
    )

    alpha_correctness_reward: float = field(
        default=1.0,
        metadata={"help": "Weight of the correctness reward function."},
    )

    sampling_mode: str = field(
        default="bernoulli",
        metadata={
            "help": "Type of sampling strategy to use. Options: ['bernoulli', 'bernoulli-argmax', 'dpls', 'categorical']."
        },
    )

    thres: float = field(
        default=0.9,
        metadata={
            "help": "Fast-dLLM confidence threshold. Used as a fixed value by remasking="
            "'fastdllm'; for remasking='block_policy' the policy chooses per block from "
            "threshold_candidates and this is ignored."
        },
    )

    dpls_stop_logit: float = field(
        default=0.0,
        metadata={
            "help": "The stop logit (utility) for DPLS sampling. Used if `sampling_mode == 'dpls'`."
        },
    )

    policy_full_context: bool = field(
        default=True,
        metadata={
            "help": "Whether to pass full sequence context to policy instead of just current block. "
            "When True, policy sees entire sequence but can only affect current block."
        },
    )

    timestep_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The batch size to use for the inner-loop policy call and log-likelihood calculations. "
            "If None, the full batch of timesteps will be processed in parallel."
        },
    )

    loglikelihood_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "If not None, loglikelihood computations and results will be in this dtype "
            "(which should be a torch floating point type such as 'float32')."
            " If None, input dtypes will be preserved (eg. bfloat16 for low-precision training)."
        },
    )

    save_best_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether to save the best checkpoint."},
    )

    es_thresholds: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Thresholds to use for expert steering (ES) rollouts."},
    )
