#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from common.models.modeling_llada import LLaDABlock
from common.models.policy_layers import RoPEDiTBlock
from common.models.policy_layers import sinusoidal_time_embedding


# The policy types whose per-position input is the dLLM's last-layer hidden state.
# generate_unified branches on this to request output_hidden_states, and the trainer
# uses it to decide whether the rollout buffer is big enough to be worth replaying.
# Keep new hidden policies here rather than repeating the tuple at each use site.
HIDDEN_POLICY_TYPES = (
    "dit_hidden",
    "dit_hidden_proj",
    "dit_block_size_hidden_proj",
)

# Of those, the ones whose rollout record can be replayed from a token state instead of
# buffered. The block_policy decode path writes its own decision-indexed rec_c buffer
# rather than going through generate_unified's per-step recorder, so replay is not
# implemented for it -- and it does not need it: it records one decision per block
# (T ~ 8) where the per-step policies record one per forward (T up to 256).
REPLAYABLE_POLICY_TYPES = (
    "dit_hidden",
    "dit_hidden_proj",
)


class PolicyConfig(PretrainedConfig):
    model_type = "policy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PolicyHFWrapper(PreTrainedModel):
    config_class = PolicyConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_policy: nn.Module,
        policy_type: str,
        config: PolicyConfig | None = None,
    ):
        super().__init__(config or PolicyConfig())
        self.base_policy = base_policy
        self.policy_type = policy_type

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self._propagate_gc_flag(True)

        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        self._propagate_gc_flag(False)

    def _propagate_gc_flag(self, value: bool):
        if hasattr(self.base_policy, "set_gradient_checkpointing"):
            self.base_policy.set_gradient_checkpointing(value)
        else:
            setattr(self.base_policy, "gradient_checkpointing", bool(value))
        for m in self.base_policy.modules():
            if hasattr(m, "set_gradient_checkpointing"):
                m.set_gradient_checkpointing(value)
            elif hasattr(m, "gradient_checkpointing"):
                setattr(m, "gradient_checkpointing", bool(value))

    @staticmethod
    def _coerce(v, dtype):
        # Only floating-point tensors get cast. Integer tensors are indices
        # (e.g. block_start) and bf16 cannot represent every integer up to the
        # generation length exactly; bool tensors are masks that downstream code
        # re-casts itself. Both must pass through untouched.
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return v.to(dtype)
        return v

    def _coerce_call(self, fn, args, kwargs):
        args = tuple(self._coerce(arg, self.dtype) for arg in args)
        kwargs = {k: self._coerce(v, self.dtype) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # coerce dtypes if needed
        return self._coerce_call(self.base_policy, args, kwargs)

    # The block_unmask decode loop calls the two heads of DiTBlockUnmaskPolicy
    # separately (block head before the block decision, unmask head after it). They
    # must go through the same dtype coercion as forward, so delegate explicitly:
    # nn.Module's __getattr__ only resolves parameters and submodules, not methods
    # of a wrapped module.
    def block_logits(self, *args, **kwargs):
        return self._coerce_call(self.base_policy.block_logits, args, kwargs)

    def unmask_logits(self, *args, **kwargs):
        return self._coerce_call(self.base_policy.unmask_logits, args, kwargs)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, *args, **kwargs):
        pass

    def tie_weights(self):
        pass


class DiTHiddenStatePolicy(nn.Module):
    def __init__(
        self,
        dllm,
        time_embed_dim: int = 128,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
    ):
        super().__init__()
        self.hidden_dim = dllm.config.hidden_size
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.mask_embedding = nn.Embedding(2, self.hidden_dim)

        # AdaLNs: since we attach an additional head (original llada block), we cannot
        # inject modulation WITHIN the block, as we do for DiTConfidencePolicy.
        # Instead, we inject scale+shift modulation (1) before each block, and (2) after the final block.
        # In the case of only having one block (as in the paper), this effectively only
        # moves one adaptation from before the FFN to after it, so it should offer similar performance.
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(num_blocks + 1)]
        )
        self.ada_linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
                for _ in range(num_blocks + 1)
            ]
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LLaDABlock.build(i, dllm.model.config, dllm.model._LLaDAModel__cache)
                for i in range(num_blocks)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_dim, 1)

        if smart_init is not None:
            self.apply_smart_init(smart_init)

    def apply_smart_init(self, target_logit: float):
        """Initialize for controlled logit distribution.

        :param target_logit: Target value for initial logit mean
        """
        with torch.no_grad():
            for linear in self.ada_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()
            self.output_proj.bias.data.fill_(target_logit)

    def forward(
        self,
        m: torch.Tensor,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param hidden_states: (*B,L,hidden_dim) hidden states from DLLM
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :return: (*B,L) unmasking logits
        """
        *B, L, D = hidden_states.shape
        assert D == self.hidden_dim, f"Expected D={self.hidden_dim}, got {D=}"
        assert m.shape == (*B, L)
        assert isinstance(timestep, torch.Tensor)
        assert timestep.shape == (*B, 1), (
            f"Unexpected {timestep.shape=}; batch dim(s) {B=}"
        )

        x = hidden_states

        # Build conditioning: time + mask
        cond = self.mask_embedding(m.int())
        time_embed = sinusoidal_time_embedding(
            timestep, self.time_embed_dim, max_period=self.time_period
        )
        time_embed = time_embed.to(cond.dtype)
        time_embed = time_embed.expand((*([-1] * len(B)), L, -1))
        time_embed = self.time_mlp(time_embed)
        cond = cond + time_embed

        original_shape = x.shape
        x_flat = x.view(-1, L, self.hidden_dim)
        cond_flat = cond.view(-1, L, self.hidden_dim)

        for i, block in enumerate(self.transformer_blocks):
            scale, bias = self.ada_linears[i](cond_flat).chunk(2, dim=-1)
            x_flat = self.norms[i](x_flat) * (1 + scale) + bias
            x_flat, _ = block(x_flat)

        scale, bias = self.ada_linears[-1](cond_flat).chunk(2, dim=-1)
        x_flat = self.norms[-1](x_flat) * (1 + scale) + bias

        x = x_flat.view(original_shape)
        raw_logits = self.output_proj(x).squeeze(-1)

        return raw_logits


class DiTConfidencePolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        feedforward_dim: int = 512,
        num_heads: int = 1,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        confidences_top_p: int = 1,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks
        self.confidences_top_p = confidences_top_p

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.confidence_proj = nn.Linear(confidences_top_p, self.hidden_dim)
        self.mask_embedding = nn.Embedding(2, self.hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                RoPEDiTBlock(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

        if smart_init is not None:
            self.apply_smart_init(smart_init)

    def apply_smart_init(self, target_logit: float):
        """Initialize for controlled logit distribution.

        Initialize mid-transformer AdaLNs to identities and sets output_proj bias to target_logit.
        This ensures initial logits are centered at target_logit, which is useful for
        DPLS sampling where you want to control the proportion of logits
        above/below the stop threshold.

        :param target_logit: Target value for initial logit mean (e.g., 0.0 to match stop_logit=0.0,
            or negative values to bias toward stopping earlier)
        """
        with torch.no_grad():
            # Initialize ada_lns to identity: x_norm * (1 + 0) + 0 = x_norm
            for block in self.transformer_blocks:
                block.ada_conditioning.weight.data.zero_()
                block.ada_conditioning.bias.data.zero_()

            # Initialize output_proj bias to target mean
            self.output_proj.bias.data.fill_(target_logit)

    def embed_input(self, c: torch.Tensor) -> torch.Tensor:
        """Map the per-position input stream to hidden_dim.

        Split out of trunk() so subclasses can swap the input representation
        without touching the transformer stack; see DiTHiddenProjPolicy.

        :param c: (*B,L,confidences_top_p) confidence values in [0,1]
        :return: (*B,L,hidden_dim)
        """
        return self.confidence_proj(c)

    def trunk(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        extra_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared backbone up to (and including) the final norm.

        Split out of forward() so that heads other than output_proj can read the
        per-position hidden states (see DiTBlockSizePolicy).

        :param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,*) per-position input stream, whatever embed_input() accepts
            (top-p confidences here, dLLM hidden states in DiTHiddenProjPolicy)
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :param extra_cond: optional (*B,L,hidden_dim) term added to the adaLN
            conditioning stream next to the mask and time embeddings; used by
            DiTBlockUnmaskPolicy to tell the unmask head where the current block is
        :return: (*B,L,hidden_dim) normalized hidden states
        """
        *B, L, _ = c.shape
        assert m.shape == (*B, L)
        assert isinstance(timestep, torch.Tensor)
        assert timestep.shape == (*B, 1), (
            f"Unexpected {timestep.shape=}; batch dim(s) {B=}"
        )

        # Conditioning: time + mask
        cond = self.mask_embedding(m.int())
        time_embed = sinusoidal_time_embedding(
            timestep, self.time_embed_dim, max_period=self.time_period
        )
        time_embed = time_embed.to(cond.dtype)
        time_embed = time_embed.expand((*([-1] * len(B)), L, -1))
        time_embed = self.time_mlp(time_embed)
        cond = cond + time_embed
        if extra_cond is not None:
            assert extra_cond.shape == cond.shape, (
                f"Unexpected {extra_cond.shape=}; expected {cond.shape}"
            )
            cond = cond + extra_cond.to(cond.dtype)

        # Embed the per-position input stream
        x = self.embed_input(c)

        # Transformer
        original_shape = x.shape
        x_flat = x.view(-1, L, self.hidden_dim)
        cond_flat = cond.view(-1, L, self.hidden_dim)
        for block in self.transformer_blocks:
            x_flat = block(x_flat, cond_flat)
        x = x_flat.view(original_shape)

        return self.final_norm(x)

    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,*) per-position input stream; see trunk()
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :return: (*B,L) unmasking logits
        """
        return self.output_proj(self.trunk(m, c, timestep)).squeeze(-1)


class HiddenProjInputMixin:
    """Swap a policy's per-position input stream from confidences to dLLM hidden states.

    Mixed in ahead of a DiTConfidencePolicy subclass so that everything from the RoPE
    DiT stack onward -- including DiTBlockSizePolicy's two heads -- is inherited
    unchanged and only embed_input() differs. A single Linear does the whole job, in
    contrast to DiTHiddenStatePolicy, which runs a full LLaDABlock at the dLLM's own
    width.

    The input LayerNorm is not optional. LLaDA's last-layer hidden states carry outlier
    channels and their norm drifts with the timestep, so without it the projection
    spends its first steps just undoing the scale.
    """

    def __init__(self, dllm, **kwargs):
        super().__init__(**kwargs)

        self.dllm_hidden_size = dllm.config.hidden_size

        # confidence_proj is the base policy's input stream and is dead here. Dropping
        # it keeps untrained weights out of every checkpoint (and out of DDP's
        # unused-parameter bookkeeping) instead of shipping them forever.
        del self.confidence_proj

        # Non-affine: this only has to fix the scale, and an affine pair would be
        # redundant with the projection immediately after it.
        self.in_norm = nn.LayerNorm(self.dllm_hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(self.dllm_hidden_size, self.hidden_dim)
        # Deliberately left at default init: zeroing it would make trunk()'s final_norm
        # see an all-zero input and kill the gradient into proj.

    def embed_input(self, h: torch.Tensor) -> torch.Tensor:
        """:param h: (*B,L,dllm_hidden_size) last-layer hidden states from the dLLM
        :return: (*B,L,hidden_dim)
        """
        assert h.shape[-1] == self.dllm_hidden_size, (
            f"Expected hidden size {self.dllm_hidden_size}, got {h.shape[-1]}"
        )
        return self.proj(self.in_norm(h))


def boundary_block_logits(
    u: torch.Tensor,
    block_start: torch.Tensor,
    candidates: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Block-size logits from per-position boundary scores.

    u[i] is read as "the score of ending the current block just after position i", so
    candidate b scores u[block_start + b - 1], plus a learnable per-candidate prior.
    Candidates that do not fit in what is left of the sequence are masked to -inf.
    Shared by DiTBlockSizePolicy and DiTBlockUnmaskPolicy.

    :param u: (*B,L) boundary scores
    :param block_start: (*B,1) integer index of the current block's first position
    :param candidates: (K,) ascending candidate block sizes
    :param bias: (K,) per-candidate prior logits
    :return: (*B,K) block-size logits with -inf on infeasible candidates
    """
    L = u.shape[-1]
    start = block_start.long()  # (*B, 1)
    end = start + candidates  # (*B, K)
    feasible = end <= L
    # Defensive: the multiple-of-8 tiling invariant means the smallest candidate
    # always fits, but an all-infeasible row would make softmax produce NaN.
    no_feasible = ~feasible.any(dim=-1, keepdim=True)
    smallest = torch.zeros_like(feasible)
    smallest[..., 0] = True
    feasible = feasible | (no_feasible & smallest)

    logits = u.gather(-1, (end - 1).clamp(max=L - 1)) + bias
    return logits.masked_fill(~feasible, float("-inf"))


def _validate_block_size_candidates(
    block_size_candidates: tuple[int, ...],
    block_size_prior_logits: tuple[float, ...] | None,
) -> torch.Tensor:
    """Check the candidate set and build the initial prior bias (zeros = uniform)."""
    if len(block_size_candidates) == 0:
        raise ValueError("block_size_candidates must be non-empty")
    if sorted(block_size_candidates) != list(block_size_candidates):
        raise ValueError(
            f"block_size_candidates must be ascending, got {block_size_candidates}"
        )
    if block_size_prior_logits is None:
        return torch.zeros(len(block_size_candidates))
    if len(block_size_prior_logits) != len(block_size_candidates):
        raise ValueError(
            f"block_size_prior_logits has {len(block_size_prior_logits)} entries "
            f"but there are {len(block_size_candidates)} candidates"
        )
    return torch.tensor(block_size_prior_logits, dtype=torch.float32)


class DiTBlockSizePolicy(DiTConfidencePolicy):
    """Joint (block size, Fast-dLLM threshold) policy; see formulation.md.

    Reuses DiTConfidencePolicy's trunk verbatim and adds two heads:

    - block size: the parent's per-position output_proj is reinterpreted as a boundary
      score, u[i] = "score of ending the current block just after position i". The logit
      of candidate b is u[block_start + b - 1] plus a learnable per-candidate prior. Only
      the K prior biases are new parameters.
    - threshold: a pooled read of the trunk's hidden states over the lookahead window
      [block_start, block_start + max_block), restricted to still-masked positions.

    The two heads are conditionally independent given the state, so the joint log-prob is
    the sum of the two. That cannot express couplings such as "large block => low
    threshold"; see formulation.md 5.3 for why v1 accepts this.
    """

    def __init__(
        self,
        block_size_candidates: tuple[int, ...] = (8, 16, 32, 64, 128),
        thresholds: tuple[float, ...] = (0.5, 0.7, 0.9),
        block_size_prior_logits: tuple[float, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if len(thresholds) == 0:
            raise ValueError("thresholds must be non-empty")
        # Per-candidate prior over block sizes. Zero => uniform at init, since
        # smart_init makes every u[i] equal and hence every gathered score equal.
        init_bias = _validate_block_size_candidates(
            block_size_candidates, block_size_prior_logits
        )

        self.register_buffer(
            "block_size_candidates",
            torch.tensor(block_size_candidates, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32),
            persistent=False,
        )
        self.block_size_bias = nn.Parameter(init_bias)

        # Zero-init so the threshold distribution also starts uniform. Note this must
        # happen here rather than in apply_smart_init, which the parent __init__ calls
        # before this head exists.
        self.threshold_head = nn.Linear(self.hidden_dim, len(thresholds))
        with torch.no_grad():
            self.threshold_head.weight.data.zero_()
            self.threshold_head.bias.data.zero_()

    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        block_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,confidences_top_p) confidence values in [0,1]
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :param block_start: (*B,1) integer index of the current block's first position
        :return: ((*B,K_block) block-size logits, (*B,K_thres) threshold logits)
        """
        *B, L, _ = c.shape
        assert block_start.shape == (*B, 1), (
            f"Unexpected {block_start.shape=}; batch dim(s) {B=}"
        )

        h = self.trunk(m, c, timestep)  # (*B, L, H)
        u = self.output_proj(h).squeeze(-1)  # (*B, L)

        start = block_start.long()  # (*B, 1)

        ### Block-size head: gather the boundary score of each candidate
        block_logits = boundary_block_logits(
            u, start, self.block_size_candidates, self.block_size_bias
        )

        ### Threshold head: masked mean-pool over the lookahead window
        max_block = int(self.block_size_candidates[-1].item())
        pos = torch.arange(L, device=h.device).expand(*B, L)  # (*B, L)
        in_window = (pos >= start) & (pos < start + max_block)  # (*B, L)
        weights = (in_window & m.bool()).to(h.dtype).unsqueeze(-1)  # (*B, L, 1)
        pooled = (h * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)
        threshold_logits = self.threshold_head(pooled)  # (*B, K_thres)

        return block_logits, threshold_logits


class DiTHiddenProjPolicy(HiddenProjInputMixin, DiTConfidencePolicy):
    """DiTConfidencePolicy reading dLLM hidden states instead of top-p confidences.

    forward(m, h, timestep) -> (*B,L) unmasking logits.
    """


class DiTBlockSizeHiddenProjPolicy(HiddenProjInputMixin, DiTBlockSizePolicy):
    """DiTBlockSizePolicy reading dLLM hidden states instead of top-p confidences.

    forward(m, h, timestep, block_start) -> (block-size logits, threshold logits).
    Both heads read the shared trunk, so swapping embed_input() is the whole change.
    """


class DiTBlockUnmaskPolicy(DiTConfidencePolicy):
    """Joint (block size, per-position unmasking) policy; formulation.md 10, item 3.

    The paper's per-position policy with a learned block schedule on top. One
    DiTConfidencePolicy trunk, two heads:

    - unmask: the parent's output_proj, unchanged. Per-position Bernoulli logits,
      sampled only inside the current block. policy_smart_init still sets its bias,
      so the initial decoding rate means what it means in the paper.
    - block size: a separate, zero-initialised boundary_proj scores "end the block
      after position i"; candidate b reads that score at block_start + b - 1 plus a
      learnable prior (boundary_block_logits). Zero init makes the block marginal
      exactly uniform over feasible candidates at step 0, unlike DiTBlockSizePolicy,
      whose reuse of output_proj only gets near-uniform.

    There is no threshold head: within a block the unmask head is the decoder, so
    Fast-dLLM's tau has nothing left to do.

    Factorisation. With ``window_cond=False`` the two heads read one trunk pass and
    are conditionally independent given the state: p(b, u | s) = p(b | s) p(u | s),
    and the sampled b only gates which positions may be drawn. With
    ``window_cond=True`` the unmask head is conditioned on the block it is decoding,
    p(b, u | s) = p(b | s) p(u | s, b): a zero-initialised ``window_embedding`` marks
    the positions in [block_start, block_end) on the adaLN conditioning stream, the
    block head reads a trunk pass *without* that term (the state it is decided in has
    no live block), and the unmask head reads a second pass *with* it. Zero init
    means step 0 is identical in both modes. The trainer sums the two log-probs per
    timestep either way; the block term is only present on the step that enters a
    new block.
    """

    def __init__(
        self,
        block_size_candidates: tuple[int, ...] = (8, 16, 32, 64, 128),
        block_size_prior_logits: tuple[float, ...] | None = None,
        window_cond: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        init_bias = _validate_block_size_candidates(
            block_size_candidates, block_size_prior_logits
        )
        self.register_buffer(
            "block_size_candidates",
            torch.tensor(block_size_candidates, dtype=torch.long),
            persistent=False,
        )
        self.block_size_bias = nn.Parameter(init_bias)
        self.window_cond = window_cond

        # Zero-init weight AND bias: every boundary score is 0, so the block logits
        # are exactly the prior. Must happen here, not in apply_smart_init, which the
        # parent __init__ runs before this head exists.
        self.boundary_proj = nn.Linear(self.hidden_dim, 1)
        with torch.no_grad():
            self.boundary_proj.weight.data.zero_()
            self.boundary_proj.bias.data.zero_()

        if window_cond:
            # Zero-init for the same reason: the windowed trunk pass starts out equal
            # to the plain one, so the unmask head's initial rate is still
            # policy_smart_init's.
            self.window_embedding = nn.Embedding(2, self.hidden_dim)
            with torch.no_grad():
                self.window_embedding.weight.data.zero_()

    @staticmethod
    def _window(
        block_start: torch.Tensor, block_end: torch.Tensor, L: int
    ) -> torch.Tensor:
        """(*B,L) bool: positions in the row's current block [block_start, block_end)."""
        positions = torch.arange(L, device=block_start.device)
        return (positions >= block_start.long()) & (positions < block_end.long())

    def block_logits(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        block_start: torch.Tensor,
    ) -> torch.Tensor:
        """Block-size head alone: trunk without the window term.

        :return: (*B,K_block) block-size logits, -inf on infeasible candidates
        """
        u = self.boundary_proj(self.trunk(m, c, timestep)).squeeze(-1)  # (*B, L)
        return boundary_block_logits(
            u, block_start, self.block_size_candidates, self.block_size_bias
        )

    def unmask_logits(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        block_start: torch.Tensor,
        block_end: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unmask head alone: trunk with the window term when window_cond is on.

        :param block_start: (*B,1) first position of the row's current block
        :param block_end: (*B,1) one past its last position; may be None when
            window_cond is off (it is not read)
        :return: (*B,L) per-position unmasking logits
        """
        extra_cond = None
        if self.window_cond:
            assert block_end is not None, (
                "window_cond=True needs block_end to condition the unmask head on"
            )
            L = c.shape[-2]
            in_block = self._window(block_start, block_end, L)  # (*B, L)
            extra_cond = self.window_embedding(in_block.long())
        h = self.trunk(m, c, timestep, extra_cond=extra_cond)
        return self.output_proj(h).squeeze(-1)

    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        block_start: torch.Tensor,
        block_end: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,confidences_top_p) confidence values in [0,1]
        :param timestep: (*B,1) tensor with diffusion timestep (in [0, 1])
        :param block_start: (*B,1) integer index of the current block's first position
        :param block_end: (*B,1) integer index one past the current block's last
            position, *after* the block decision of this step; required when
            window_cond is on, ignored otherwise
        :return: ((*B,L) unmasking logits, (*B,K_block) block-size logits)
        """
        *B, L, _ = c.shape
        assert block_start.shape == (*B, 1), (
            f"Unexpected {block_start.shape=}; batch dim(s) {B=}"
        )
        if block_end is not None:
            assert block_end.shape == (*B, 1), (
                f"Unexpected {block_end.shape=}; batch dim(s) {B=}"
            )

        if not self.window_cond:
            # One pass, two heads: the recorded and replayed numerics of the
            # conditionally independent policy are unchanged.
            h = self.trunk(m, c, timestep)  # (*B, L, H)
            unmask_logits = self.output_proj(h).squeeze(-1)  # (*B, L)
            u = self.boundary_proj(h).squeeze(-1)  # (*B, L)
            block_logits = boundary_block_logits(
                u, block_start, self.block_size_candidates, self.block_size_bias
            )
            return unmask_logits, block_logits

        return (
            self.unmask_logits(m, c, timestep, block_start, block_end),
            self.block_logits(m, c, timestep, block_start),
        )
