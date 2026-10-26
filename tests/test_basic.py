import numpy as np
from src import config as C, data_gen, fairness, scene_cut, annotations
from src.models import Tiny3DCNN, focal_loss
import torch


def test_clip_shape_and_range():
    clip, _ = data_gen.make_clip(1, 0, np.random.default_rng(0))
    assert clip.shape == (C.N_FRAMES, C.FRAME_SIZE, C.FRAME_SIZE)
    assert 0.0 <= clip.min() and clip.max() <= 1.0


def test_train_set_is_biased_test_is_balanced():
    _, m = data_gen.make_dataset(400, True, 1)
    p = m.filter(m["label"] == 0)["group"].to_numpy().mean()      # share of dark_bg for class 0
    assert p < 0.2
    _, m2 = data_gen.make_dataset(400, False, 1)
    assert abs(m2["group"].to_numpy().mean() - 0.5) < 0.1


def test_reweighing_makes_class_and_group_independent():
    _, m = data_gen.make_dataset(800, True, 2)
    y, g = m["label"].to_numpy(), m["group"].to_numpy()
    w = fairness.reweighing_weights(y, g)
    for c in range(4):
        share = w[(y == c) & (g == 0)].sum() / w[y == c].sum()
        assert abs(share - (g == 0).mean()) < 1e-3


def test_group_report_perfect_and_gap():
    y = np.array([0, 1, 0, 1]); g = np.array([0, 0, 1, 1])
    assert fairness.group_report(y, y, g)["worst_group"] == 1.0
    bad = np.array([0, 1, 1, 0])
    r = fairness.group_report(y, bad, g)
    assert r["gap"] == 1.0 and r["worst_group"] == 0.0


def test_scene_cut_f1_with_tuned_threshold():
    fv, tv = data_gen.make_stream(40, 4)            # validation stream -> tune k
    k = scene_cut.tune_k(fv, tv)
    f, t = data_gen.make_stream(40, 3)              # separate test stream
    pr, rc, f1 = scene_cut.prf(scene_cut.detect(f, k), t)
    assert pr > 0.7 and f1 > 0.6


def test_majority_vote_beats_worst_annotator():
    y = np.random.default_rng(0).integers(0, 4, 600)
    _, s, _, _ = annotations.analyse(annotations.simulate(y, 1))
    assert s["majority_accuracy"] > s["annotator_accuracy"]["ann_C"]


def test_cnn_forward_and_focal_loss():
    m = Tiny3DCNN(); x = torch.randn(2, 1, C.N_FRAMES, C.FRAME_SIZE, C.FRAME_SIZE)
    out = m(x); assert out.shape == (2, 4)
    l = focal_loss(out, torch.tensor([0, 1])); assert (l >= 0).all()
