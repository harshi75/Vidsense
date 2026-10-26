"""Simulated multi-annotator labels and a Polars-based annotation analysis.

Three annotators label every clip; each has its own error rate, and errors are
more likely between *similar* motion classes (horizontal<->vertical,
static<->others). We turn the raw annotations into insights:
majority-vote label, agreement rate, Cohen's kappa, per-class noise, and a
list of clips that should be sent back for re-annotation.
"""
import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score

ANNOTATOR_ERR = {"ann_A": 0.04, "ann_B": 0.10, "ann_C": 0.20}
CONFUSABLE = {0: [1, 2], 1: [2, 3], 2: [1, 3], 3: [1, 2]}


def simulate(labels, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for name, err in ANNOTATOR_ERR.items():
        out = labels.copy()
        flip = rng.random(len(labels)) < err
        for i in np.where(flip)[0]:
            out[i] = rng.choice(CONFUSABLE[int(labels[i])])
        cols[name] = out
    return pl.DataFrame({"clip_idx": np.arange(len(labels)), "true_label": labels, **cols})


def analyse(df):
    ann = list(ANNOTATOR_ERR)
    df = df.with_columns(
        pl.concat_list(ann).alias("votes")
    ).with_columns(
        pl.col("votes").list.eval(pl.element().mode().first()).list.first().alias("majority"),
        pl.col("votes").list.n_unique().alias("n_distinct"),
    ).with_columns(
        (pl.col("n_distinct") == 1).alias("unanimous"),
        (pl.col("n_distinct") == 3).alias("all_disagree"),
    )
    summary = {
        "unanimous_rate": df["unanimous"].mean(),
        "all_disagree_rate": df["all_disagree"].mean(),
        "majority_accuracy": (df["majority"] == df["true_label"]).mean(),
        "annotator_accuracy": {a: float((df[a] == df["true_label"]).mean()) for a in ann},
        "kappa": {f"{a}-{b}": float(cohen_kappa_score(df[a], df[b]))
                  for i, a in enumerate(ann) for b in ann[i + 1:]},
    }
    per_class = (df.group_by("true_label")
                   .agg((pl.col("majority") != pl.col("true_label")).mean().alias("majority_error"),
                        pl.col("unanimous").mean().alias("unanimous_rate"),
                        pl.len().alias("n"))
                   .sort("true_label"))
    review = df.filter(~pl.col("unanimous")).select("clip_idx", "votes", "majority")
    return df, summary, per_class, review
