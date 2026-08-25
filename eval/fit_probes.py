"""Fit the Stage 0 probes and report the go/no-go.

Three logistic probes on the dataset from eval.collect_probe_data, all predicting the
same label ("was it safe to unmask this position at this step?"):

    top-1 conf   -> what every policy in this repo sees today
    top-16 conf  -> one config line away (confidences_top_p: 16)
    hidden(4096) -> what the projector policies were built for

The split is by *example*, not by row: rows from one generation share a trajectory and
are heavily correlated, so a random row split would leak and flatter every probe.

Reading the result:

    hidden ~= top-1     the observation is not the bottleneck; drop the whole line and
                        look at the reward / credit assignment instead
    top-16 ~= hidden    set confidences_top_p: 16 and skip the projector entirely
    hidden >  both      the hidden input has something, and a dense-label warm-start
                        has signal to teach it

Limitation worth keeping in mind: this measures linear separability of a *proxy* for
the RL objective. A gap does not guarantee an RL gain, and no gap does not strictly
rule one out, since the real policy reads the input through a DiT rather than a linear
map. If the answer comes out ambiguous, the next step is to fit the actual policy trunk
on each input rather than a linear probe.

Example:
    python -m eval.fit_probes --data /work/hdd/bhta/zsun9/probe_data/gsm8k_t09.npz
"""

import argparse

import numpy as np
import torch


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64, device=scores.device)
    # average ranks within tied groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    pos = labels.bool()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit(x_tr, y_tr, x_te, y_te, epochs=60, lr=0.05, wd=1e-4, device="cuda"):
    """Logistic regression with standardised inputs, full-batch LBFGS-free training."""
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp(min=1e-6)
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    w = torch.zeros(x_tr.shape[1], 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    # class weighting so a skewed positive rate does not collapse the fit
    pos_w = ((y_tr == 0).sum() / (y_tr == 1).sum().clamp(min=1)).clamp(0.05, 20.0)
    for _ in range(epochs):
        opt.zero_grad()
        logits = (x_tr @ w).squeeze(-1) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tr, pos_weight=pos_w
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        return auc((x_te @ w).squeeze(-1) + b, y_te), float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    d = np.load(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ex = d["example_id"]
    uniq = np.unique(ex)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    n_te = max(1, int(len(uniq) * args.test_frac))
    te_ex = set(uniq[:n_te].tolist())
    te = np.array([e in te_ex for e in ex])
    tr = ~te

    y = torch.as_tensor(d["label"], dtype=torch.float32, device=device)
    print(f"rows: {len(y)}  train {int(tr.sum())} / test {int(te.sum())}  "
          f"examples {len(uniq)} (test {n_te})")
    print(f"positive rate: train {float(y[tr].mean()):.4f}  test {float(y[te].mean()):.4f}")
    print(f"{'probe':<16} {'dim':>6} {'held-out AUC':>14}")

    conf = torch.as_tensor(d["conf"], dtype=torch.float32, device=device)
    feature_sets = {
        "top-1 conf": conf[:, :1],
        "top-16 conf": conf,
        "hidden (4096)": torch.as_tensor(d["h"], dtype=torch.float32, device=device),
    }
    results = {}
    for name, x in feature_sets.items():
        a, _ = fit(x[tr], y[tr], x[te], y[te], epochs=args.epochs, device=device)
        results[name] = a
        print(f"{name:<16} {x.shape[1]:>6} {a:>14.4f}")

    print()
    h, t1, t16 = results["hidden (4096)"], results["top-1 conf"], results["top-16 conf"]
    print(f"hidden - top-1  = {h - t1:+.4f}")
    print(f"hidden - top-16 = {h - t16:+.4f}")
    print(f"top-16 - top-1  = {t16 - t1:+.4f}")


if __name__ == "__main__":
    main()
