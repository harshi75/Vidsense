"""Bias / fairness utilities: reweighing + group-wise metrics."""
import numpy as np


def reweighing_weights(labels, groups):
    """Kamiran & Calders (2012) reweighing: w(c,g) = P(c) P(g) / P(c,g).
    After weighting, class and group become statistically independent."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    w = np.ones(len(labels), dtype=np.float32)
    for c in np.unique(labels):
        for g in np.unique(groups):
            m = (labels == c) & (groups == g)
            if m.sum():
                w[m] = (labels == c).mean() * (groups == g).mean() / m.mean()
    return w


def group_report(y, pred, groups):
    """Accuracy per group, worst-group accuracy, gap and macro TPR gap
    (equal-opportunity difference averaged over classes)."""
    y, pred, groups = map(np.asarray, (y, pred, groups))
    acc = {int(g): float((pred[groups == g] == y[groups == g]).mean()) for g in np.unique(groups)}
    tpr_gaps = []
    for c in np.unique(y):
        t = [(pred[(y == c) & (groups == g)] == c).mean() for g in np.unique(groups)]
        tpr_gaps.append(max(t) - min(t))
    return dict(overall=float((pred == y).mean()), group_acc=acc,
                worst_group=min(acc.values()), gap=max(acc.values()) - min(acc.values()),
                eo_gap=float(np.mean(tpr_gaps)))


def paired_bootstrap(y, pred_a, pred_b, groups, n=2000, seed=0):
    """Offline A/B-style check: bootstrap CI of the worst-group-accuracy
    difference (B - A) on the same test clips."""
    rng = np.random.default_rng(seed)
    y, pa, pb, g = map(np.asarray, (y, pred_a, pred_b, groups))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        ra, rb = group_report(y[idx], pa[idx], g[idx]), group_report(y[idx], pb[idx], g[idx])
        diffs.append(rb["worst_group"] - ra["worst_group"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)
