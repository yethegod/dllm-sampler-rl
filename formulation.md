# Learning Joint Block-Size and Threshold Policies for Diffusion LLMs

*A categorical policy over (block size, confidence threshold), trained with GRPO, following the
setup of "Learning Unmasking Policies for Diffusion Language Models" (arXiv:2512.09106).*

---

## 1. Motivation

A masked diffusion LLM $p_\phi$ decodes a length-$L$ generation region semi-autoregressively:
the region is split into consecutive **blocks**, blocks are decoded left to right, and positions
*within* a block are unmasked in parallel over several denoising steps. Fast-dLLM decides which
positions to unmask by a confidence threshold $\tau$: at each step, every still-masked position
in the current block whose top-1 probability exceeds $\tau$ is unmasked at once.

Two hyperparameters govern the compute/accuracy trade-off: the block length $B$ and the
threshold $\tau$. The paper this repo forks holds both fixed and learns *which positions* to
unmask. This document learns the two schedule knobs themselves.

The original version of this document assumed a specific story — small blocks are accurate and
slow, large blocks are fast and inaccurate — and proposed learning block size alone.
**A full sweep (§2) showed that story is wrong**, and the design here is a direct consequence of
what was measured rather than of what was assumed.

---

## 2. Empirical grounding

Fast-dLLM constants on GSM8K (all 1319 questions), LLaDA-8B-Instruct, $L = 256$, greedy
($T=0$), seed 42. Complete 5×3 grid. Source:
`/work/hdd/bhta/zsun9/eval_results/blocksweep/summary_statistics.csv`.

**accuracy % / NFE**

| $b$ | $\tau=0.5$ | $\tau=0.7$ | $\tau=0.9$ |
|---|---|---|---|
| 8   | 70.81 / 57.0 | 78.54 / 73.5 | 79.08 / 99.8 |
| 16  | 69.67 / 44.2 | 78.39 / 59.7 | 79.68 / 88.1 |
| 32  | 63.08 / 37.3 | 77.71 / 52.0 | 79.45 / 81.5 |
| 64  | **65.81 / 32.6** | **79.61 / 47.2** | **80.67 / 78.0** |
| 128 | 61.18 / 28.7 | 72.71 / 44.5 | 71.11 / 81.3 |

Three facts drive every design choice below.

**(a) At fixed $\tau$, block size buys almost no compute.** At $\tau = 0.9$, $b = 64$ *strictly
dominates* all four other block sizes — highest accuracy **and** lowest NFE. There is no dial to
turn. At $\tau = 0.7$, $b = 64$ dominates $b \in \{8,16,32\}$, leaving only $b = 128$ (6% cheaper,
6.9 points worse). At $\tau = 0.5$, $b = 64$ again dominates $b = 32$ on both axes. The only
genuine within-column trade-offs are $b \in \{8, 16, 64, 128\}$ at $\tau = 0.5$ — and that whole
column is far below the frontier set by $\tau = 0.7$.

**(b) Accuracy vs. block size is $\wedge$-shaped, peaking at 64 in every column — not
monotone.** Small blocks are both worse *and* slower. Likely mechanism: a block cannot advance
until every position in it is filled, so small blocks force `argmax` unmasking of low-confidence
positions (`common/generation/generation.py`, the `if not unmask_local.any()` fallback), whereas
a large block lets the decoder spend its steps on confident positions elsewhere and revisit the
hard ones with more context.

Note what this costs the action space: of the five block candidates, **$b = 8$ and $b = 32$ never
appear on the global Pareto frontier at any $\tau$**. The policy's first job is simply to learn
not to pick them.

**(c) $\tau$ is the compute lever, roughly 2× stronger than $b$.** Within $\tau = 0.9$, NFE spans
78–100 (28%). Across $\tau$ at $b = 64$, NFE spans 47–78 (65%), and $\tau = 0.5$ reaches 37.

### 2.1 The exact bar

Because every constant policy in the action space has been measured, the best *state-independent*
policy is known exactly rather than guessed. Per-example
$R_i = \mathbb{1}[\hat y_i = y_i]\cdot((L - \min(N_i,L) + 1)/L)^{\alpha}$:

| $\alpha$ | best constant | $\mathbb{E}[R]$ | per-example oracle |
|---|---|---|---|
| 0 | $(b{=}64,\ \tau{=}0.9)$ | 0.8067 | 0.9325 (+15.6%) |
| 0.25 | $(b{=}64,\ \tau{=}0.7)$ | 0.7590 | 0.9032 (+19.0%) |
| **0.5** | $(b{=}64,\ \tau{=}0.7)$ | **0.7238** | 0.8750 (+20.9%) |
| 1.0 | $(b{=}64,\ \tau{=}0.7)$ | 0.6587 | 0.8218 (+24.7%) |

**$b = 64$ is optimal at every $\alpha$; $\tau$ flips once near $\alpha \approx 0.083$ and then
stays at 0.7.** Two consequences:

- $\alpha > 0$ is still required (at $\alpha = 0$ the optimum is the slowest accurate action), but
  **sweeping $\alpha$ will not trace out a frontier of learned policies**, because the optimum
  barely moves. The $\alpha$ sweep proposed in the original document is dropped; $\alpha = 0.5$ is
  the primary setting and $\alpha \in \{0.25, 1.0\}$ are sensitivity checks only.
- The only route to beating a constant is **genuine state-dependence**. That, and nothing else,
  is what this project tests.

### 2.2 How much state-dependence is even available

Since all 15 constants were run on the same questions with the same seed, per-example outcomes
can be compared directly:

| | |
|---|---|
| best constant $(64, 0.7)$ solves | **80.7%** |
| at least one of the 15 actions solves | **93.3%** |
| all 15 actions solve | 23.3% |
| solved by some but not all ("disagreement band") | **70.0%** |

An oracle choosing the best action per example scores $\mathbb{E}[R] = 0.8750$ at $\alpha = 0.5$
vs. 0.7238 for the best constant — **+20.9%**.

**This is a loose upper bound and must not be read as achievable headroom.** Decoding is
deterministic at $T = 0$, so the flips are not sampling noise; they are chaotic sensitivity to
the unmasking schedule — change the order slightly and the answer cascades elsewhere. Taking a
max over 15 such binary outcomes harvests a large amount of variation that is unpredictable *in
principle* from the block-start state. The project's bet, stated precisely:

> **Some non-trivial fraction of the 70% disagreement band is predictable from the confidence
> field observed at block boundaries.**

Nothing cheaper than training answers this. §9 states what counts as a refutation.

---

## 3. Setup and notation

| Symbol | Meaning |
|---|---|
| $p_\phi$ | frozen diffusion LLM (LLaDA-8B-Instruct); never updated |
| $q$ | prompt token ids |
| $L$ | generation length (256) |
| $\mathbb{M}$ | mask token id (126336 for LLaDA) |
| $\mathcal{B} = \{8,16,32,64,128\}$ | candidate block sizes, $\lvert\mathcal{B}\rvert = 5$ |
| $\mathcal{T} = \{0.5, 0.7, 0.9\}$ | candidate thresholds, $\lvert\mathcal{T}\rvert = 3$ |
| $\mathcal{A} = \mathcal{B} \times \mathcal{T}$ | joint action space, 15 actions |
| $P$ | width of the top-$p$ confidence feature (`confidences_top_p`, default 1) |

$\mathcal{T}$ is exactly the grid swept in §2, so **every action has a measured constant-policy
baseline** — a deliberate choice that makes §9's falsification test exact.

A rollout produces block starts $p_0 = 0$, $p_{t+1} = p_t + b_t$, terminating at $p_T = L$.

**Feasibility.** Only $b$ is constrained: $\mathcal{B}_t = \{b \in \mathcal{B} : b \le L - p_t\}$;
$\tau$ is always free. Every candidate is a multiple of 8 and $L = 256$ is a multiple of 8, so
$L - p_t$ stays a multiple of 8, $\mathcal{B}_t$ is never empty, and blocks tile $[0,L)$ exactly
with no partial tail. Hence $T \in [L/128,\ L/8] = [2, 32]$.

---

## 4. The MDP

**Timing.** One decision at the **first forward pass of each block**. That same forward also
performs the block's first unmasking step, so the decision costs **zero additional NFE**. A
rollout therefore has tens of forward passes but only $T \in [2,32]$ decisions.

### State

At decision $t$, one forward of the frozen dLLM on the current partially-decoded sequence $x_t$
yields $\text{probs} = \mathrm{softmax}(p_\phi(x_t))$ over the generation region, and

$$s_t = (c_t,\ m_t,\ \nu_t,\ p_t)$$

- $c_t \in [0,1]^{L\times P}$ — top-$P$ confidences at **every** generation position, including
  positions far past the frontier that are still fully masked. This is the signal the policy is
  meant to exploit.
- $m_t \in \{0,1\}^L$ — mask indicator, 1 = still masked.
- $\nu_t = n_t / L$ — normalized NFE consumed so far (the repo's `steps_taken / L`).
- $p_t$ — current block start; recoverable from $m_t$, but passed explicitly because the
  block-size head indexes by it.

### Action

$$a_t = (b_t,\ \tau_t) \in \mathcal{B}_t \times \mathcal{T}$$

### Transition

Deterministic given $p_\phi$ and the greedy setting. Fast-dLLM runs on $[p_t,\ p_t + b_t)$ **with
the chosen $\tau_t$**:

> repeat: let $\gamma_i = \max_v \text{probs}[i,v]$; unmask every still-masked $i$ in the block
> with $\gamma_i > \tau_t$; if none qualifies, force-unmask the argmax; re-run the forward —
> until the block has no masked positions.

This costs $n_t \in [1, b_t]$ forwards (the first being the shared decision forward). Then
$p_{t+1} = p_t + b_t$.

### Reward

Terminal only, $\gamma = 1$, $r_t = 0$ for $t < T-1$:

$$R = \mathbb{1}[\hat y = y]\cdot c_{\text{pos}} \cdot \left(\frac{L - \min(N,L) + 1}{L}\right)^{\alpha},
\qquad N = \sum_{t=0}^{T-1} n_t$$

This is the repo's existing `_multiplicative_step_scaling_reward_func` unchanged, with $N$ =
total NFE = `steps_taken`, $c_{\text{pos}}$ = `alpha_correctness_reward` = 1.0, $\alpha$ =
`alpha_compute_reward` = 0.5. Multiplicative gating means a wrong answer scores 0 regardless of
speed, so the policy cannot win by being fast and wrong.

---

## 5. Policy

$\pi_\theta(b, \tau \mid s_t)$ reuses the paper's `DiTConfidencePolicy` trunk **unchanged**:

$$
\begin{aligned}
\text{cond} &= \mathrm{MaskEmb}(m_t) + \mathrm{TimeMLP}(\mathrm{SinEmb}(\nu_t)) &&\in \mathbb{R}^{L\times H}\\
h^{(0)} &= \mathrm{ConfProj}(c_t) &&\in \mathbb{R}^{L\times H}\\
h^{(\ell+1)} &= \mathrm{RoPEDiTBlock}(h^{(\ell)},\ \text{cond}) &&\ell = 0,\dots,n_{\text{blk}}-1\\
h &= \mathrm{LayerNorm}(h^{(n_{\text{blk}})}) &&\in \mathbb{R}^{L\times H}\\
u &= \mathrm{OutputProj}(h) \in \mathbb{R}^{L} &&(H\to 1,\ \text{squeezed})
\end{aligned}
$$

Everything through $u$ is byte-identical to `common/models/policy.py`, including RoPE over
absolute generation positions (`policy_full_context = true`, so index 0 always means "start of
generation region"). Two heads read off it.

### 5.1 Block-size head (boundary scoring)

Read $u[i]$ as *"the score of ending the current block immediately after position $i$"*:

$$\ell^{b}_t[k] = u[\,p_t + b_k - 1\,] + \beta_k,\qquad \beta \in \mathbb{R}^{5}\ \text{learnable}$$
$$\ell^{b}_t[k] \leftarrow -\infty\quad\text{if } b_k > L - p_t$$

Zero architectural change — the only new parameters are the 5 prior biases $\beta$. Candidate-set
agnostic: changing $\mathcal{B}$ changes which indices are gathered, not the network.

### 5.2 Threshold head (pooled)

$\tau$ is not tied to a position, so it reads a pooled summary of the lookahead window
$W_t = [p_t,\ \min(p_t + b_{\max},\ L))$:

$$\bar h_t = \frac{\sum_{i \in W_t} m_t[i]\, h[i]}{\sum_{i\in W_t} m_t[i]},
\qquad \ell^{\tau}_t = \mathrm{Linear}(H \to 3)(\bar h_t)$$

### 5.3 Factorization

**v1 assumes conditional independence given the state:**

$$\pi_\theta(b,\tau \mid s_t) = \pi_\theta(b \mid s_t)\cdot \pi_\theta(\tau \mid s_t),
\qquad \log\pi_\theta(a_t\mid s_t) = \log\pi^{b}_\theta + \log\pi^{\tau}_\theta$$

with both softmaxes taken at `temperature_policy` (1.0 in training).

**Known expressiveness limit.** Independent heads cannot represent a coupling such as *"if I take
a large block, use a lower threshold"* — they will place mass on cross terms like
$(128, 0.9)$ even if only $(64, 0.7)$ and $(128, 0.5)$ are good. §2 shows $b$ and $\tau$ do
interact (the dominance structure differs by column). The conditional form
$\pi_\theta(\tau \mid s_t, b_t)$ — feed a $b$-embedding into the threshold head and sample $b$
first — is the designated ablation (§10), deferred to keep v1 simple.

### 5.4 Initialization

Measured, not assumed (the first draft of this section got it wrong):

- **Threshold head: exactly uniform.** Its `Linear(H→3)` is zero-initialized in both weight and
  bias, so $\ell^{\tau} = 0$ regardless of state.
- **Block-size head: *near*-uniform, with a mild state-dependent tilt.** `smart_init(c)` sets
  `output_proj.bias = c` and zeros the AdaLN modulators, but it does **not** touch
  `output_proj.weight`, which stays randomly initialized. So $u[i] = W h[i] + c$ still varies
  across positions, and the gathered candidate scores are not equal. Measured at init with
  $\beta = 0$: $\pi_\theta(b) \approx (0.165, 0.164, 0.222, 0.221, 0.228)$, entropy 1.599 nats
  vs. $\log 5 = 1.609$.

This is harmless — arguably useful, since it gives GRPO a non-degenerate starting gradient — but
it means the untrained policy is *close to* rather than *identical to* the uniform-random
baseline, which weakens one of the §9 checks accordingly.

Note `policy_smart_init` no longer controls the initial decoding rate the way it does in the
paper; $\beta$ and the $\tau$ head bias do.

New parameters over the paper's policy: **392 total** — 5 for $\beta$, 387 for the threshold head
($128\times3$ weight + 3 bias) — against a ~331k-parameter trunk.

---

## 6. Training objective (GRPO)

For each prompt, sample $G$ rollouts from $\pi_\theta$; they share the prompt but differ in their
$(b,\tau)$ sequences, hence in both $N$ and $\hat y$.

**Advantages** — mean-centred only, no division by std (Dr.GRPO style, matching the repo):

$$A_g = R_g - \frac1G\sum_{g'} R_{g'}$$

**Clipped surrogate**, where the "token" is one block-level decision:

$$\rho_{g,t} = \frac{\pi_\theta(b_{g,t},\tau_{g,t}\mid s_{g,t})}{\pi_{\text{old}}(b_{g,t},\tau_{g,t}\mid s_{g,t})}
= \exp\big[(\log\pi^b_\theta + \log\pi^\tau_\theta) - (\log\pi^b_{\text{old}} + \log\pi^\tau_{\text{old}})\big]$$

$$\mathcal{J}(\theta) = \frac1G\sum_g \frac{1}{T_g}\sum_{t=0}^{T_g-1}
\min\big(\rho_{g,t}A_g,\ \mathrm{clip}(\rho_{g,t},1-\epsilon,1+\epsilon)A_g\big),\qquad \mathcal{L} = -\mathcal{J}$$

with $\epsilon = 0.5$ and **no KL term** ($\beta_{\text{KL}} = 0$, asserted by the trainer).
$\pi_{\text{old}}$ is the behavior policy; `num_iterations = 2` reuses each rollout batch for two
gradient steps, so $\rho \ne 1$ on the second.

The scalar $A_g$ is broadcast across all $T_g$ decisions — no finer credit assignment.

**Known bias in $1/T_g$.** Rollouts choosing many small blocks make more decisions, each
down-weighted by $1/T_g$; a 2-block rollout gets 16× the per-decision gradient of a 32-block one.
Inherited verbatim from the paper's implementation (`train/trainer.py`, divide by
`num_active_steps`). Kept for parity; $(\sum_g T_g)^{-1}\sum_g\sum_t(\cdot)$ is a one-line
alternative and a natural ablation.

---

## 7. Credit-assignment characteristics

| | Paper (unmasking policy) | This work |
|---|---|---|
| decisions per rollout | $T\times L$ Bernoulli bits, $T\approx 30\text{–}100$ | $T$ joint categorical draws, $T\in[2,32]$ |
| action entropy at init | $L\cdot H(\mathrm{Bern}(\sigma(-2)))$ per step | $\le \log 15 \approx 2.71$ nats per block |
| reward signal | 1 scalar per rollout | 1 scalar per rollout |

The action space is orders of magnitude smaller, so gradient variance per decision is higher.
Mitigations, in order:

1. **Larger groups.** `num_generations` 8 → 16 (config-only; keeps
   `per_device_train_batch_size = 16`, one group per GPU, `generation_batch_size = 128` on 8 GPUs).
2. **Monitor collapse** — log the 5-way and 3-way histograms and $H(\pi_\theta)$ every step
   (§9). The trainer already skips steps where all advantages are ~0, so a collapsed policy stops
   learning rather than diverging.
3. **Optional entropy bonus** (not in v1; the trainer already computes entropy for logging).

---

## 8. Baselines and evaluation

GSM8K, $L = 256$, same prompts and seeds throughout. Metrics: accuracy and mean NFE, plus
$\mathbb{E}[R]$ at $\alpha = 0.5$ (the trained objective).

| # | Baseline | Status |
|---|---|---|
| 0 | **The measured constant grid** $\mathcal{B}\times\mathcal{T}$ (§2) — every action as a fixed policy | **done** (13/15) |
| 1 | Best constant: $(b{=}64,\tau{=}0.7)$, 80.7% acc, $\mathbb{E}[R] = 0.7238$ | **done** — this is the bar |
| 2 | Constant-policy Pareto frontier: $(0.5,32), (0.5,16), (0.7,128), (0.7,64), (0.9,64)$ | **done** |
| 3 | AdaBlock heuristic (`--adaptive_block --delimiter_threshold 0.3`) | to run |
| 4 | Random action $\sim\mathrm{Uniform}(\mathcal{B}_t\times\mathcal{T})$ | to run |
| 5 | Untrained policy ($\theta$ at init) — must match #4 by §5.4 | to run |
| 6 | **Learned policy** | to run |

**Claim to be tested:** the learned policy beats the best constant *in its own action space* —
$\mathbb{E}[R] > 0.7238$ at $\alpha = 0.5$. This is stronger and more honest than "beats
fixed-block decoding", and it is exactly measurable because of §2.

**Diagnostics** recorded per example: the realized $(b_t,\tau_t)$ sequence; the correlation
between each choice and the mean confidence in its lookahead window (the direct test of
state-dependence); the $T$ and $N$ distributions; whether choices drift systematically over the
course of a generation.

**Aside, independent of this project.** The repo's standing Fast-dLLM baseline, $\tau{=}0.9$ at
$B{=}32$ (79.45% @ 81.5 NFE), is dominated by $\tau{=}0.7$ at $B{=}64$ (79.61% @ 47.2 NFE) —
same accuracy for **42% fewer NFE**. Every Fast-dLLM number previously reported in this repo was
measured at a dominated operating point and should be restated.

---

## 9. What would falsify this

Stated in advance so the result is interpretable either way.

**Early stop (within a few hundred training steps).** If $H(\pi_\theta)$ collapses and the action
histogram converges to a point mass on $(64, 0.7)$, the policy has learned the marginal — i.e.
the constant we already have. Stop; do not run a full evaluation.

**Negative result.** If the trained policy's $\mathbb{E}[R] \le 0.7238$ at $\alpha = 0.5$, or its
accuracy $\le 80.7\%$ at comparable NFE, then per-example schedule adaptation does not help on
this task. Given §2.2, that is a plausible outcome — the disagreement band may be chaotic rather
than predictable — and it is a legitimate finding worth reporting, not a bug to be tuned away.

**Implementation checks.** These test the code, not the hypothesis — a failure here means a bug,
not a negative result:

- An untrained policy should land *near* the uniform-random-action baseline. Per §5.4 the block
  head starts near-uniform rather than exactly uniform, so this is a sanity band, not an equality.
- The threshold marginal at init must be exactly $(\tfrac13,\tfrac13,\tfrac13)$.
- No infeasible action may ever be sampled, and $\sum_t b_t = L$ exactly for every rollout.
- Forcing $\beta$ to a one-hot and the $\tau$ head bias to a one-hot must reproduce the
  corresponding fixed-$(b,\tau)$ Fast-dLLM run token-for-token — the strongest available check
  that the new decoding path is equivalent to the existing one.

**Weak-positive trap.** A policy that beats 0.7238 but whose action choices show ~zero
correlation with the confidence field has found a better *marginal* (e.g. a lucky mixture), not
state-dependence. Report the correlation alongside the score.

---

## 10. Scope and extensions

**v1:** joint $(b,\tau)$ per block, conditionally independent heads, Fast-dLLM within blocks.

**Extensions, in order of expected value:**

1. **Conditional factorization** $\pi_\theta(\tau\mid s_t, b_t)$ — removes the §5.3 limitation;
   the measured $b\times\tau$ interaction makes this the most likely real gain.
2. Finer $\mathcal{T}$ (e.g. adding 0.6, 0.8) once the coarse grid shows signal. Note each new
   value costs one constant-baseline eval to keep §9's bar exact.
3. Replace the within-block decoder with the paper's learned per-position policy — the shared
   trunk makes this a three-head model and the GRPO loss simply sums the action log-probs.
4. Let the block boundary be an arbitrary position rather than a member of $\mathcal{B}$ — the
   boundary-scoring head already computes the required per-position scores.

---

## 11. Implementation notes

- **Decoding** — a new branch in `common/generation/generation.py:generate_unified`, selected by
  `remasking="block_policy"`. Unlike the existing fixed-block and AdaBlock paths, block
  boundaries must be **per row** (`(B,L)` block masks rather than a shared `slice`): group
  members necessarily choose different block sizes, and training rolls out 128 sequences at a
  time. The existing AdaBlock path sidesteps this by requiring batch size 1, which is not an
  option here.
- **Per-row $\tau$ is already supported.** `_confidence_threshold_unmask` types `thres` as
  `float | torch.Tensor` and compares `(B,BL) > thres`, so a `(B,1)` tensor broadcasts per row —
  Expert Steering already relies on this. No change to the comparison is needed.
- **Policy** — `DiTBlockSizePolicy` subclassing `DiTConfidencePolicy`, adding $\beta$ and the
  threshold head; `forward(m, c, timestep, block_start) -> (logits_b, logits_tau)`. The trunk must
  expose $h$ (pre-`output_proj`) for the pooled head.
  `PolicyHFWrapper.forward` casts every tensor arg to `self.dtype`; `block_start` is an integer
  index and must be exempted (bf16 cannot represent all integers up to 256 exactly), so restrict
  the cast to floating-point tensors.
- **Sampling** — `categorical_sample` / `categorical_batch_loglik` / `categorical_entropy` in
  `common/generation/sampling.py`, alongside `bernoulli_*` and `dpls_*`.
- **Trainer** — `compute_loss` is unchanged. Recorded rollout tensors keep the shapes the
  existing machinery expects, with the two action streams stored side by side:
  `sampling_masks (B,T,1)`, `samples (B,T,2)`, `sampling_inputs (B,T,5+3)`, and `policy_inputs`
  as a 4-tuple sliced along the time axis.
