"""VidSense – Streamlit demo.

Shows the saved results of the full pipeline (results/metrics.json) and lets you try the
fast parts live: clip generation, optical-flow features, the lighting-shortcut demo,
scene-cut detection and annotation analysis. PyTorch training is NOT run here
(it takes ~12 minutes); run `python -m src.run_all` locally for that.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import annotations, config as C, data_gen, fairness, features, scene_cut   # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#003B8B", "#E8702A", "#2E8B57", "#D64545", "#8A8F98"
CLASSES = C.CLASSES
st.set_page_config(page_title="VidSense", page_icon="🎞️", layout="wide")


@st.cache_data
def load_metrics():
    return json.loads((C.RESULTS_DIR / "metrics.json").read_text())


@st.cache_data(show_spinner="Generating clips and extracting optical flow…")
def shortcut_data(seed: int):
    Xtr, mtr = data_gen.make_dataset(400, True, 100 + seed)      # biased training set
    Xte, mte = data_gen.make_dataset(240, False, 200 + seed)     # balanced test set
    return (features.batch_features(Xtr), mtr["label"].to_numpy(), Xtr.mean((1, 2, 3)),
            features.batch_features(Xte), mte["label"].to_numpy(), mte["group"].to_numpy(), Xte.mean((1, 2, 3)))


def upscale(frame, k=4):
    return np.kron(frame, np.ones((k, k), dtype=np.float32))


M = load_metrics()
b, f, r = M["baseline_cnn"], M["focal_cnn"], M["reweighed_cnn"]

st.title("🎞️ VidSense")
st.caption("Bias-aware video motion analysis · Harshita Singh · NITRA Technical Campus · "
           "all data are synthetic")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results", "Clip lab", "Shortcut demo", "Scene cuts", "Annotations"])

# ------------------------------------------------------------------ 1. results
with tab1:
    st.subheader("Saved results of the full pipeline")
    st.write("A 3D-CNN is trained on videos where lighting is tied to the motion class (about 90 %), "
             "then tested on a balanced set. Numbers are the mean of 3 seeds from `results/metrics.json`.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Worst-group accuracy", f"{r['worst_group']*100:.1f} %", f"{(r['worst_group']-b['worst_group'])*100:+.1f} pts vs baseline")
    c2.metric("Equal-opportunity gap", f"{r['eo_gap']:.3f}", f"{r['eo_gap']-b['eo_gap']:+.3f} vs baseline", delta_color="inverse")
    c3.metric("Overall accuracy", f"{r['overall']*100:.1f} %", f"{(r['overall']-b['overall'])*100:+.1f} pts vs baseline")
    df = pd.DataFrame([
        ["RF + flow features", M["rf_flow"]["overall"], M["rf_flow"]["worst_group"], M["rf_flow"]["eo_gap"]],
        ["RF + flow + brightness (shortcut)", M["rf_flow_plus_brightness"]["overall"], M["rf_flow_plus_brightness"]["worst_group"], M["rf_flow_plus_brightness"]["eo_gap"]],
        ["CNN baseline (cross-entropy)", b["overall"], b["worst_group"], b["eo_gap"]],
        ["CNN focal loss", f["overall"], f["worst_group"], f["eo_gap"]],
        ["CNN reweighed", r["overall"], r["worst_group"], r["eo_gap"]]],
        columns=["Model", "Overall acc.", "Worst-group acc.", "EO gap"])
    st.dataframe(df.style.format({"Overall acc.": "{:.1%}", "Worst-group acc.": "{:.1%}", "EO gap": "{:.3f}"}), hide_index=True, width="stretch")
    ab = M["ab_worst_group_diff"]
    st.info(f"Paired bootstrap (seed 0): reweighing improves worst-group accuracy by {ab['mean']*100:.1f} points, "
            f"95 % CI [{ab['ci_low']*100:.1f}, {ab['ci_high']*100:.1f}].")
    fig_dir = C.RESULTS_DIR / "figures"
    g1, g2 = st.columns(2)
    g1.image(str(fig_dir / "group_bars.png"), caption="Overall vs worst-group accuracy")
    g2.image(str(fig_dir / "heatmaps.png"), caption="Where the baseline fails (seed 0)")
    st.caption(f"Summary ({M['llm_summary']['source']}): {M['llm_summary']['text']}")

# ------------------------------------------------------------------ 2. clip lab
with tab2:
    st.subheader("Generate a clip and look at its motion features")
    a, b_ = st.columns(2)
    cls = a.selectbox("Motion class", CLASSES, index=3)
    light = b_.radio("Lighting", ["bright background", "dark background"], horizontal=True)
    seed = st.slider("Random seed", 0, 99, 7)
    clip, meta = data_gen.make_clip(CLASSES.index(cls), 0 if light.startswith("bright") else 1, np.random.default_rng(seed))
    idx = [0, 3, 6, 9, 12, 15]
    cols = st.columns(len(idx))
    for col, t in zip(cols, idx):
        col.image(upscale(clip[t]), caption=f"t={t}", clamp=True, width="stretch")
    fv = features.clip_features(clip)
    st.write("**Optical-flow features** (Farnebäck) – these do not depend on lighting polarity:")
    st.dataframe(pd.DataFrame({"feature": features.FLOW_FEATURES, "value": np.round(fv, 3)}), hide_index=True, height=280)
    st.caption(f"Object speed {meta['speed']:.2f} px/frame · mean brightness {clip.mean():.2f}")

# ------------------------------------------------------------------ 3. shortcut demo
with tab3:
    st.subheader("Watch a model pick up a shortcut")
    st.write("A Random Forest is trained on 400 clips where bright backgrounds go with static/horizontal motion "
             "and dark ones with vertical/circular motion, then tested on 240 balanced clips. "
             "Adding **mean brightness** as a feature lets it use the shortcut.")
    s = st.slider("Data seed", 0, 20, 0, key="sc_seed")
    Ftr, ytr, btr, Fte, yte, gte, bte = shortcut_data(s)
    from sklearn.ensemble import RandomForestClassifier
    rf0 = RandomForestClassifier(150, random_state=0).fit(Ftr, ytr)
    rf1 = RandomForestClassifier(150, random_state=0).fit(np.c_[Ftr, btr], ytr)
    p0, p1 = rf0.predict(Fte), rf1.predict(np.c_[Fte, bte])
    r0, r1 = fairness.group_report(yte, p0, gte), fairness.group_report(yte, p1, gte)
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); x = np.arange(2)
    ax.bar(x - .2, [r0["group_acc"][0] * 100, r0["group_acc"][1] * 100], .4, color=BLUE, label="flow only")
    ax.bar(x + .2, [r1["group_acc"][0] * 100, r1["group_acc"][1] * 100], .4, color=ORANGE, label="flow + brightness")
    ax.set_xticks(x); ax.set_xticklabels(["bright background", "dark background"]); ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    m1, m2 = st.columns(2)
    m1.metric("Flow only: worst group", f"{r0['worst_group']*100:.1f} %", f"EO gap {r0['eo_gap']:.3f}", delta_color="off")
    m2.metric("With brightness: worst group", f"{r1['worst_group']*100:.1f} %", f"EO gap {r1['eo_gap']:.3f}", delta_color="off")
    st.caption("Results vary with the seed – that variability is itself one of the findings of the project.")

# ------------------------------------------------------------------ 4. scene cuts
with tab4:
    st.subheader("Scene-cut detection")
    n = st.slider("Clips in the stream", 10, 60, 24)
    k = st.slider("Threshold k (median + k × MAD)", 2.0, 15.0, 5.0, 0.5)
    sd = st.slider("Stream seed", 0, 50, 3, key="cut_seed")
    frames, truth = data_gen.make_stream(n, 1000 + sd)
    scores = scene_cut.cut_scores(frames)
    pred = scene_cut.detect(frames, k)
    pr, rc, f1 = scene_cut.prf(pred, truth)
    med = np.median(scores); mad = np.median(np.abs(scores - med))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(np.arange(1, len(scores) + 1), scores, color=BLUE, lw=1.2)
    ax.axhline(med + k * mad, color=ORANGE, ls="--", label="threshold")
    for i, t in enumerate(truth): ax.axvline(t, color=GREEN, alpha=.35, label="true cut" if i == 0 else None)
    ax.scatter(pred, scores[pred - 1], color=RED, zorder=3, s=22, label="detected")
    ax.set_xlabel("frame"); ax.set_ylabel("change score"); ax.legend(frameon=False, ncol=3); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{pr:.2f}"); c2.metric("Recall", f"{rc:.2f}"); c3.metric("F1", f"{f1:.2f}")
    st.caption("Lower k finds more cuts but adds false alarms. Cuts between clips with the same lighting are the hard ones.")

# ------------------------------------------------------------------ 5. annotations
with tab5:
    st.subheader("Three annotators, one set of labels")
    ann = pl.read_csv(C.RESULTS_DIR / "annotations.csv")
    A = M["annotations"]["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Majority-vote accuracy", f"{A['majority_accuracy']*100:.1f} %")
    c2.metric("Unanimous clips", f"{A['unanimous_rate']*100:.1f} %")
    c3.metric("Flagged for review", f"{M['annotations']['n_flagged']}")
    acc = A["annotator_accuracy"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(acc) + ["majority"], [v * 100 for v in acc.values()] + [A["majority_accuracy"] * 100], color=[BLUE, "#7FA6DB", ORANGE, GREEN])
    ax.set_ylim(60, 100); ax.set_ylabel("label accuracy (%)"); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.write("**Pairwise Cohen's kappa:**", {k: round(v, 2) for k, v in A["kappa"].items()})
    only = st.checkbox("Show only clips where annotators disagree", value=True)
    view = ann.filter(~((pl.col("ann_A") == pl.col("ann_B")) & (pl.col("ann_B") == pl.col("ann_C")))) if only else ann
    st.dataframe(view.head(200).to_pandas(), hide_index=True, height=300)
    st.caption(f"{view.height} clips shown (first 200). Labels: 0 static, 1 horizontal, 2 vertical, 3 circular.")
