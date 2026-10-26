"""Create all figures used in the report / slides from results/metrics.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from . import config as C

BLUE, LBLUE, ORANGE, GREY, GREEN = "#003B8B", "#7FA6DB", "#E8702A", "#8A8F98", "#2E8B57"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 100})
FIG = C.RESULTS_DIR / "figures"; FIG.mkdir(exist_ok=True, parents=True)
R = json.load(open(C.RESULTS_DIR / "metrics.json"))
CL = C.CLASSES


def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=200); plt.close()


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.4)
    boxes = [("Synthetic\nvideo data", .2), ("Parquet +\nPolars", 2.0), ("Features\n(optical flow)", 3.8),
             ("Models\nsklearn / 3D-CNN", 5.6), ("Bias check &\nmitigation", 7.4), ("Report +\nLLM summary", 9.2)]
    for i, (t, x) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.6, 1.3, boxstyle="round,pad=0.05", fc=BLUE if i % 2 == 0 else LBLUE, ec="none"))
        ax.text(x + .8, 1.65, t, ha="center", va="center", color="white" if i % 2 == 0 else "black", fontsize=10.5, fontweight="bold")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.78, 1.65), xytext=(x + 1.62, 1.65), arrowprops=dict(arrowstyle="-|>", lw=2, color=ORANGE, shrinkA=0, shrinkB=0))
    ax.text(5.5, 0.35, "Side branches: scene-cut detection (unsupervised) · KMeans clustering · annotation analysis",
            ha="center", fontsize=10, color=GREY)
    save("pipeline.png")


def samples():
    X = np.load(C.DATA_DIR / "sample_clips.npy")
    import polars as pl
    m = pl.read_parquet(C.DATA_DIR / "metadata.parquet").filter(pl.col("split") == "test").head(16)
    lab, grp = m["label"].to_numpy(), m["group"].to_numpy()
    fig, axs = plt.subplots(4, 6, figsize=(10, 6.8))
    for r, c in enumerate(range(4)):
        idx = np.where(lab == c)[0][0]
        for k, t in enumerate([0, 3, 6, 9, 12, 15]):
            axs[r, k].imshow(X[idx, t], cmap="gray", vmin=0, vmax=1); axs[r, k].axis("off")
            if r == 0: axs[r, k].set_title(f"t={t}")
        axs[r, 0].text(-6, 16, CL[c], rotation=90, va="center", ha="right", fontsize=12, fontweight="bold")
    save("samples.png")


def bias_bar():
    p = {d["label"]: d["p_bright"] for d in R["train_p_bright_by_class"]}
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); x = np.arange(4)
    ax.bar(x - .2, [p[i] * 100 for i in range(4)], .4, color=BLUE, label="Train / val (biased)")
    ax.bar(x + .2, [50] * 4, .4, color=LBLUE, label="Test (balanced)")
    ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_ylabel("% clips with bright background"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.15), ncol=2); save("bias_bar.png")


def curves():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for n, c in zip(["baseline_cnn", "focal_cnn", "reweighed_cnn"], [BLUE, ORANGE, GREEN]):
        h = R["train_history"][n]
        axs[0].plot([e["epoch"] for e in h], [e["train_loss"] for e in h], color=c, lw=2, label=n)
        axs[1].plot([e["epoch"] for e in h], [e["val_acc"] * 100 for e in h], color=c, lw=2, label=n)
    axs[0].set_title("Training loss"); axs[1].set_title("Validation accuracy (%)")
    for a in axs: a.set_xlabel("Epoch")
    axs[0].legend(frameon=False); save("curves.png")


def group_bars():
    names = [("rf_flow", "RF + flow"), ("baseline_cnn", "CNN\nbaseline"), ("focal_cnn", "CNN\nfocal"), ("reweighed_cnn", "CNN\nreweighed")]
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(names))
    ov = [R[n]["overall"] * 100 for n, _ in names]; wg = [R[n]["worst_group"] * 100 for n, _ in names]
    ov_e = [R[n].get("overall_std", 0) * 100 for n, _ in names]; wg_e = [R[n].get("worst_group_std", 0) * 100 for n, _ in names]
    ax.bar(x - .2, ov, .4, yerr=ov_e, color=BLUE, label="Overall accuracy", capsize=3)
    ax.bar(x + .2, wg, .4, yerr=wg_e, color=ORANGE, label="Worst-group accuracy", capsize=3)
    for i in range(len(names)):
        ax.text(x[i] - .2, ov[i] + 1.5, f"{ov[i]:.1f}", ha="center", fontsize=9); ax.text(x[i] + .2, wg[i] + 1.5, f"{wg[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in names]); ax.set_ylim(0, 112); ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center"); save("group_bars.png")


def heatmaps():
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for a, key, t in zip(axs, ["class_group_acc_baseline", "class_group_acc_reweighed"], ["Baseline CNN (seed 0)", "Reweighed CNN (seed 0)"]):
        M = np.array(R[key]) * 100
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=100)
        a.set_xticks([0, 1]); a.set_xticklabels(["bright bg", "dark bg"]); a.set_yticks(range(4)); a.set_yticklabels(CL); a.set_title(t)
        for i in range(4):
            for j in range(2): a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="white" if M[i, j] > 55 else "black", fontweight="bold")
    save("heatmaps.png")


def confusion():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
    for a, key, t in zip(axs, ["confusion_baseline", "confusion_reweighed"], ["Baseline CNN", "Reweighed CNN"]):
        M = np.array(R[key]); a.imshow(M, cmap="Blues")
        a.set_xticks(range(4)); a.set_xticklabels(CL, rotation=30); a.set_yticks(range(4)); a.set_yticklabels(CL)
        a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(t)
        for i in range(4):
            for j in range(4): a.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    save("confusion.png")


def scene():
    d = R["scene_cut_demo"]; s = np.array(d["scores"]); med = np.median(s); mad = np.median(np.abs(s - med))
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.plot(np.arange(1, len(s) + 1), s, color=BLUE, lw=1.3)
    ax.axhline(med + d["k"] * mad, color=ORANGE, ls="--", label=f"threshold (k={d['k']:.0f})")
    for i, t in enumerate(d["truth"]): ax.axvline(t, color=GREEN, alpha=.35, lw=1.5, label="true cut" if i == 0 else None)
    ax.set_xlabel("Frame index"); ax.set_ylabel("Change score"); ax.legend(frameon=False); save("scene_cut.png")


def cluster():
    c = R["clustering"]; names = list(c); lab = ["Flow\nfeatures", "CNN emb.\nbaseline", "CNN emb.\nreweighed"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); x = np.arange(3)
    ax.bar(x - .2, [c[n]["ari_vs_motion"] for n in names], .4, color=BLUE, label="ARI vs motion class (want high)")
    ax.bar(x + .2, [c[n]["ari_vs_lighting"] for n in names], .4, color=ORANGE, label="ARI vs lighting group (want low)")
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("Adjusted Rand Index"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, 1.25)); save("cluster.png")


def annot():
    a = R["annotations"]["summary"]; fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    ac = a["annotator_accuracy"]; axs[0].bar(list(ac), [v * 100 for v in ac.values()], color=[BLUE, LBLUE, ORANGE])
    axs[0].bar(["majority"], [a["majority_accuracy"] * 100], color=GREEN); axs[0].set_ylim(60, 100); axs[0].set_title("Label accuracy (%)")
    k = a["kappa"]; axs[1].bar(list(k), list(k.values()), color=LBLUE); axs[1].set_ylim(0, 1); axs[1].set_title("Cohen's kappa")
    axs[1].tick_params(axis="x", labelsize=8); save("annotations.png")


def importances():
    imp = sorted(R["rf_importances"].items(), key=lambda t: t[1])[-8:]
    fig, ax = plt.subplots(figsize=(6, 3.8)); ax.barh([k for k, _ in imp], [v for _, v in imp], color=BLUE); ax.set_xlabel("RF importance"); save("importances.png")


if __name__ == "__main__":
    for f in (pipeline, samples, bias_bar, curves, group_bars, heatmaps, confusion, scene, cluster, annot, importances):
        f(); print("ok", f.__name__)
