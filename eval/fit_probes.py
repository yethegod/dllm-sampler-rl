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


def fit(x_tr, y_tr, x_te, y_te, epochs=400, lr=0.05, wd=1e-4, hidden=0, device="cuda"):
    """Logistic (hidden=0) or one-hidden-layer MLP probe on standardised inputs.

    Returns (test AUC, train AUC, final loss). The train AUC is not decoration: a
    4096-d probe scoring below a 1-d one is only meaningful if it actually converged,
    and train ~= test ~= low is the signature that separates "not linearly accessible"
    from "the optimiser never got there".
    """
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp(min=1e-6)
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    if hidden:
        net = torch.nn.Sequential(
            torch.nn.Linear(x_tr.shape[1], hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        ).to(device)
    else:
        net = torch.nn.Linear(x_tr.shape[1], 1).to(device)
        torch.nn.init.zeros_(net.weight)
        torch.nn.init.zeros_(net.bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # class weighting so a skewed positive rate does not collapse the fit
    pos_w = ((y_tr == 0).sum() / (y_tr == 1).sum().clamp(min=1)).clamp(0.05, 20.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(x_tr).squeeze(-1), y_tr, pos_weight=pos_w
        )
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return (
            auc(net(x_te).squeeze(-1), y_te),
            auc(net(x_tr).squeeze(-1), y_tr),
            float(loss),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--wd_grid", type=float, nargs="+", default=[1e-4, 1e-2, 1e-1, 1.0, 10.0]
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=128,
        help="Also probe h after a PCA projection to this many dims, fitted on the "
        "training rows only. This is not just regularisation: HiddenProjInputMixin "
        "applies Linear(4096 -> policy_hidden_dim), so this is the representation the "
        "policy actually receives, and it makes the capacity comparison against the "
        "16-d confidences fair. 0 disables.",
    )
    parser.add_argument(
        "--mlp_hidden",
        type=int,
        default=128,
        help="Also fit a one-hidden-layer probe on the hidden state. The top-1 "
        "confidence is itself a nonlinear readout of h (max over softmax(W h)), which "
        "a linear probe cannot represent, so a linear-only comparison is biased "
        "against the hidden input. 0 disables.",
    )
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
    # sub-train / validation split, again by example
    tr_ex = [e for e in uniq if e not in te_ex]
    n_va = max(1, int(len(tr_ex) * 0.25))
    va_ex = set(tr_ex[:n_va])
    va = np.array([e in va_ex for e in ex])
    su = tr & ~va
    print(f"wd selection: sub-train {int(su.sum())} / val {int(va.sum())} rows "
          f"({len(tr_ex) - n_va} / {n_va} examples)")
    print(f"{'probe':<22} {'dim':>6} {'test AUC':>10} {'train AUC':>10} "
          f"{'val AUC':>9} {'wd':>8}")

    conf = torch.as_tensor(d["conf"], dtype=torch.float32, device=device)
    h = torch.as_tensor(d["h"], dtype=torch.float32, device=device)
    feature_sets = [
        ("top-1 conf", conf[:, :1], 0),
        ("top-16 conf", conf, 0),
        ("hidden (4096)", h, 0),
    ]
    if args.proj_dim:
        # PCA fitted on training rows only -- fitting it on everything would leak the
        # test examples' covariance structure into the projection.
        xt = h[tr]
        mu = xt.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(xt - mu, q=args.proj_dim, niter=4)
        h_proj = (h - mu) @ v
        feature_sets.append((f"hidden PCA-{args.proj_dim}", h_proj, 0))
        if args.mlp_hidden:
            feature_sets.append(
                (f"hidden PCA-{args.proj_dim} + MLP", h_proj, args.mlp_hidden)
            )
    if args.mlp_hidden:
        feature_sets += [
            ("hidden + MLP", h, args.mlp_hidden),
            ("top-16 + MLP", conf, args.mlp_hidden),
        ]
    results = {}
    for name, x, hid in feature_sets:
        # Pick weight decay on held-out EXAMPLES carved out of train, never on test.
        # Fixing it at 1e-4 is what made the first run's numbers a function of how long
        # the optimiser happened to run rather than of the representation.
        best = (-1.0, None, None, None)
        for wd in args.wd_grid:
            a_val, _, _ = fit(
                x[su], y[su], x[va], y[va], epochs=args.epochs, wd=wd,
                hidden=hid, device=device,
            )
            if a_val > best[0]:
                best = (a_val, wd, None, None)
        wd = best[1]
        a, a_tr, _ = fit(
            x[tr], y[tr], x[te], y[te], epochs=args.epochs, wd=wd,
            hidden=hid, device=device,
        )
        results[name] = a
        print(
            f"{name:<22} {x.shape[1]:>6} {a:>10.4f} {a_tr:>10.4f} "
            f"{best[0]:>9.4f} {wd:>8.0e}"
        )

    print()
    a_h = results["hidden (4096)"]
    a_t1 = results["top-1 conf"]
    a_t16 = results["top-16 conf"]
    print(f"hidden - top-1  = {a_h - a_t1:+.4f}")
    print(f"hidden - top-16 = {a_h - a_t16:+.4f}")
    print(f"top-16 - top-1  = {a_t16 - a_t1:+.4f}")
    if "hidden + MLP" in results:
        print(f"hidden+MLP - top-1 = {results['hidden + MLP'] - a_t1:+.4f}")


if __name__ == "__main__":
    main()
