# VidSense – Bias-Aware Video Motion Analysis Pipeline (Mini Project)

A small, fully reproducible mini project on **shortcut learning in video classification**:
it shows how a video model can rely on lighting instead of motion, measures the damage, and
reduces it. It covers video analysis (optical flow, 3D-CNN, scene cuts), supervised and
unsupervised learning, loss functions, bias mitigation, PyTorch + scikit-learn, Polars/Arrow data
handling, annotation analysis, offline evaluation (A/B-style) and an LLM-assisted summary.

**Author:** Harshita Singh (B.Tech CES-AIML, NITRA Technical Campus, Ghaziabad) · Mentor: Mr. Shahanawaz

## Headline results (3 seeds, balanced test set)
| Model | Overall % | Worst-group % | EO gap |
|---|---|---|---|
| CNN baseline (cross-entropy) | 86.3 | 81.4 | 0.288 |
| CNN focal loss | 87.4 | 86.1 | 0.281 |
| CNN reweighed | 97.4 | 97.1 | 0.033 |

All data are synthetic; conclusions concern the method, not absolute accuracy on real video.

## What it does
1. **Generates synthetic video clips** (16 frames, 32x32) of a blob moving in 4 ways
   (static / horizontal / vertical / circular). Lighting (bright vs dark background) is a
   nuisance attribute that is *deliberately correlated* with the class in train/val (~90 %)
   and balanced in test, to create a dataset bias.
2. Stores metadata as **Parquet** and analyses it with **Polars** (lazy API).
3. Extracts **optical-flow features** (OpenCV Farneback) -> Random Forest / Logistic Regression.
4. Trains a **PyTorch 3D-CNN** with three setups: cross-entropy, focal loss, and
   cross-entropy + **reweighing** (Kamiran & Calders) for bias mitigation. 3 seeds each.
5. **Unsupervised**: KMeans on CNN embeddings / flow features; **scene-cut detection** on long streams.
6. **Annotation analysis**: 3 simulated annotators -> agreement, Cohen's kappa, noisy-clip review list.
7. **Offline verification**: worst-group accuracy, equal-opportunity gap, paired bootstrap CI.
8. **LLM summary** (optional): set `ANTHROPIC_API_KEY` (and optionally `LLM_MODEL`); otherwise a template is used.

## Run
```bash
pip install -r requirements.txt
python -m src.run_all        # ~13 min on 1 CPU core, writes results/metrics.json (resumable)
python -m src.make_figures   # writes results/figures/*.png
```

## Layout
```
src/config.py        settings           src/models.py      3D-CNN + CE / focal loss
src/data_gen.py      synthetic videos   src/fairness.py    reweighing + group metrics
src/features.py      optical-flow feats src/scene_cut.py   scene-cut detector
src/annotations.py   annotator analysis src/llm_report.py  LLM / template summary
src/run_all.py       full pipeline      src/make_figures.py plots
```

## Scaling note (PySpark)
Metadata analysis uses Polars. The same group-by in PySpark would be
`df.groupBy("split","label","group").agg(F.count("*"), F.avg("speed"))`. 
