"""End-to-end VidSense pipeline.  Usage:  python -m src.run_all

Steps
 1. Generate biased train/val and balanced test videos        (data_gen)
 2. Store metadata as Parquet and analyse it with Polars       (Polars + Arrow)
 3. Classical baseline: optical-flow features + sklearn        (supervised)
 4. PyTorch 3D-CNN: baseline CE / focal / reweighed            (loss + bias mitigation)
 5. Unsupervised: KMeans on CNN embeddings & flow features
 6. Scene-cut detection on long streams
 7. Annotation analysis (3 annotators)
 8. Offline A/B-style bootstrap comparison, LLM summary, save JSON + plots
"""
import json, pickle, time
import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, f1_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from . import config as C, data_gen, features, scene_cut, annotations, fairness, llm_report
from .models import Tiny3DCNN, loss_fn

torch.set_num_threads(1)


def to_tensor(X):
    return torch.from_numpy(X).unsqueeze(1)          # (N,1,T,H,W)


def train_cnn(Xtr, ytr, wtr, Xva, yva, loss_name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Tiny3DCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, C.LR, total_steps=C.EPOCHS * int(np.ceil(len(Xtr) / C.BATCH_SIZE)))
    lf = loss_fn(loss_name)
    Xt, yt, wt = to_tensor(Xtr), torch.from_numpy(ytr.copy()), torch.from_numpy(wtr)
    hist = []
    for ep in range(C.EPOCHS):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(Xt), C.BATCH_SIZE):
            b = perm[i:i + C.BATCH_SIZE]
            opt.zero_grad(); loss = lf(model(Xt[b]), yt[b], wt[b]); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        va = (predict(model, Xva) == yva).mean()
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(Xt), val_acc=float(va)))
    return model, hist


@torch.no_grad()
def predict(model, X, embed=False):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        xb = to_tensor(X[i:i + 128])
        out.append((model.embed(xb) if embed else model(xb).argmax(1)).numpy())
    return np.concatenate(out)


def main():
    t0 = time.time(); R = {}
    C.DATA_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)

    # 1 -- data ------------------------------------------------------------------
    Xtr, mtr = data_gen.make_dataset(C.N_TRAIN, True, 1000)
    Xva, mva = data_gen.make_dataset(C.N_VAL, True, 2000)
    Xte, mte = data_gen.make_dataset(C.N_TEST, False, 3000)
    ytr, yva, yte = (m["label"].to_numpy() for m in (mtr, mva, mte))
    gtr, gva, gte = (m["group"].to_numpy() for m in (mtr, mva, mte))

    # 2 -- Parquet + Polars ------------------------------------------------------
    meta = pl.concat([mtr.with_columns(pl.lit("train").alias("split")),
                      mva.with_columns(pl.lit("val").alias("split")),
                      mte.with_columns(pl.lit("test").alias("split"))])
    meta.write_parquet(C.DATA_DIR / "metadata.parquet")
    lazy = (pl.scan_parquet(C.DATA_DIR / "metadata.parquet")
            .group_by("split", "label", "group").agg(pl.len().alias("n"), pl.col("speed").mean().alias("mean_speed"))
            .sort("split", "label", "group").collect())
    R["dataset_table"] = lazy.to_dicts()
    cg = (meta.filter(pl.col("split") == "train").group_by("label")
          .agg((pl.col("group") == 0).mean().alias("p_bright")).sort("label"))
    R["train_p_bright_by_class"] = cg.to_dicts()
    np.save(C.DATA_DIR / "sample_clips.npy", Xte[:16])     # tiny sample for the figures

    # 3 -- classical features + sklearn -------------------------------------------
    Ftr, Fva, Fte = (features.batch_features(X) for X in (Xtr, Xva, Xte))
    sc = StandardScaler().fit(Ftr)
    rf = RandomForestClassifier(200, random_state=0).fit(Ftr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Ftr), ytr)
    brightness = lambda X: X.mean((1, 2, 3))[:, None]            # shortcut feature
    rf_b = RandomForestClassifier(200, random_state=0).fit(np.c_[Ftr, brightness(Xtr)], ytr)
    R["rf_flow"] = fairness.group_report(yte, rf.predict(Fte), gte)
    R["logreg_flow"] = fairness.group_report(yte, lr.predict(sc.transform(Fte)), gte)
    R["rf_flow_plus_brightness"] = fairness.group_report(yte, rf_b.predict(np.c_[Fte, brightness(Xte)]), gte)
    R["rf_importances"] = dict(zip(features.FLOW_FEATURES, map(float, rf.feature_importances_)))

    # 4 -- CNN variants over 3 seeds ---------------------------------------------
    wr = fairness.reweighing_weights(ytr, gtr).astype(np.float32)
    ones = np.ones(len(ytr), dtype=np.float32)
    variants = {"baseline_cnn": ("ce", ones), "focal_cnn": ("focal", ones), "reweighed_cnn": ("ce", wr)}
    runs = {k: [] for k in variants}; preds = {k: [] for k in variants}; hists = {}
    best_models = {}
    for name, (lname, w) in variants.items():
        for s in C.SEEDS:
            t = time.time()
            cache = C.RESULTS_DIR / "cache" / f"{name}_{s}.pkl"
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():                       # resume support
                rep, p, h, state = pickle.load(open(cache, "rb"))
                model = Tiny3DCNN(); model.load_state_dict(state)
            else:
                model, h = train_cnn(Xtr, ytr, w, Xva, yva, lname, s)
                p = predict(model, Xte)
                rep = fairness.group_report(yte, p, gte); rep["macro_f1"] = float(f1_score(yte, p, average="macro"))
                pickle.dump((rep, p, h, model.state_dict()), open(cache, "wb"))
            runs[name].append(rep); preds[name].append(p)
            if s == C.SEEDS[0]:
                hists[name] = h; best_models[name] = model
            print(f"{name} seed={s} acc={rep['overall']:.3f} worst={rep['worst_group']:.3f} ({time.time()-t:.0f}s)", flush=True)
    def agg(lst, key):
        v = np.array([r[key] for r in lst]); return float(v.mean()), float(v.std())
    for name in variants:
        R[name] = {k: agg(runs[name], k)[0] for k in ("overall", "worst_group", "gap", "eo_gap", "macro_f1")}
        R[name].update({k + "_std": agg(runs[name], k)[1] for k in ("overall", "worst_group", "gap", "eo_gap", "macro_f1")})
        R[name]["group_acc"] = {g: float(np.mean([r["group_acc"][g] for r in runs[name]])) for g in (0, 1)}
        R[name]["per_seed"] = runs[name]
    R["train_history"] = hists
    R["confusion_baseline"] = confusion_matrix(yte, preds["baseline_cnn"][0]).tolist()
    R["confusion_reweighed"] = confusion_matrix(yte, preds["reweighed_cnn"][0]).tolist()
    # per class x group accuracy, seed 0, for the heat-map
    def cg_acc(p):
        return [[float((p[(yte == c) & (gte == g)] == c).mean()) for g in (0, 1)] for c in range(4)]
    R["class_group_acc_baseline"], R["class_group_acc_reweighed"] = cg_acc(preds["baseline_cnn"][0]), cg_acc(preds["reweighed_cnn"][0])

    # 5 -- unsupervised -----------------------------------------------------------
    emb = predict(best_models["reweighed_cnn"], Xte, embed=True)
    emb_b = predict(best_models["baseline_cnn"], Xte, embed=True)
    Fs = StandardScaler().fit_transform(Fte)
    R["clustering"] = {}
    for nm, Z in {"flow_features": Fs, "cnn_embedding_baseline": emb_b, "cnn_embedding_reweighed": emb}.items():
        km = KMeans(4, n_init=10, random_state=0).fit(Z)
        R["clustering"][nm] = dict(ari_vs_motion=float(adjusted_rand_score(yte, km.labels_)),
                                   ari_vs_lighting=float(adjusted_rand_score(gte, km.labels_)),
                                   silhouette=float(silhouette_score(Z, km.labels_)))

    # 6 -- scene cuts -------------------------------------------------------------
    sc_rows = []
    for s in range(5):
        fv, tv = data_gen.make_stream(60, 500 + s); k = scene_cut.tune_k(fv, tv)
        ft, tt = data_gen.make_stream(60, 600 + s); p = scene_cut.detect(ft, k)
        pr, rc, f1 = scene_cut.prf(p, tt)
        sc_rows.append(dict(stream=s, k=k, precision=pr, recall=rc, f1=f1, n_true=len(tt), n_pred=len(p)))
    R["scene_cut"] = dict(rows=sc_rows, mean={m: float(np.mean([r[m] for r in sc_rows])) for m in ("precision", "recall", "f1")})
    fs, ts = data_gen.make_stream(24, 777); R["scene_cut_demo"] = dict(scores=scene_cut.cut_scores(fs).tolist(), truth=ts.tolist(), k=sc_rows[0]["k"])

    # 7 -- annotations ------------------------------------------------------------
    adf = annotations.simulate(ytr, seed=5)
    adf, asum, apc, review = annotations.analyse(adf)
    R["annotations"] = dict(summary=asum, per_class=apc.to_dicts(), n_flagged=review.height)
    adf.select("clip_idx", "true_label", "ann_A", "ann_B", "ann_C", "majority").write_csv(C.RESULTS_DIR / "annotations.csv")
    # effect of label noise on the classical model
    noisy = {a: RandomForestClassifier(200, random_state=0).fit(Ftr, adf[a].to_numpy()).predict(Fte) for a in annotations.ANNOTATOR_ERR}
    noisy["majority_vote"] = RandomForestClassifier(200, random_state=0).fit(Ftr, adf["majority"].to_numpy()).predict(Fte)
    R["label_noise_effect"] = {k: float((v == yte).mean()) for k, v in noisy.items()}

    # 8 -- offline A/B-style comparison + LLM summary --------------------------------
    m, lo, hi = fairness.paired_bootstrap(yte, preds["baseline_cnn"][0], preds["reweighed_cnn"][0], gte)
    R["ab_worst_group_diff"] = dict(mean=m, ci_low=lo, ci_high=hi)
    R["llm_summary"] = llm_report.summarise({k: R[k] for k in ("baseline_cnn", "reweighed_cnn")})
    R["runtime_sec"] = time.time() - t0
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("done in %.0fs" % R["runtime_sec"])


if __name__ == "__main__":
    main()
