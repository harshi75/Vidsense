"""Synthetic video generator.

Each clip shows one Gaussian blob moving with one of four motion patterns
(static / horizontal / vertical / circular). The *lighting group* decides
whether the blob is dark-on-bright or bright-on-dark.

In train/val the lighting group is deliberately correlated with the class
(a "spurious correlation", i.e. a dataset bias). The test set is balanced.
A model that learns the shortcut ("bright background => static/horizontal")
will do well on train but badly on minority groups at test time.
"""
import numpy as np
import polars as pl
from . import config as C


def _blob(xc, yc, size, sigma=2.0):
    ys, xs = np.mgrid[0:size, 0:size]
    return np.exp(-((xs - xc) ** 2 + (ys - yc) ** 2) / (2 * sigma ** 2))


def make_clip(cls, group, rng):
    """Return a (T, H, W) float32 clip in [0, 1] and a dict of metadata."""
    S, T = C.FRAME_SIZE, C.N_FRAMES
    if group == 0:   # bright background, dark object
        bg, fg = rng.uniform(0.70, 0.80), rng.uniform(0.15, 0.25)
    else:            # dark background, bright object
        bg, fg = rng.uniform(0.15, 0.25), rng.uniform(0.75, 0.85)

    speed = rng.uniform(0.8, 1.3)
    sign = rng.choice([-1, 1])
    x0, y0 = rng.uniform(10, S - 10), rng.uniform(10, S - 10)
    cx, cy, r = rng.uniform(13, S - 13), rng.uniform(13, S - 13), rng.uniform(6, 9)
    phase = rng.uniform(0, 2 * np.pi)
    omega = sign * rng.uniform(0.30, 0.40)

    frames = np.empty((T, S, S), dtype=np.float32)
    for t in range(T):
        if cls == 0:       # static (tiny jitter)
            x, y = x0 + rng.normal(0, 0.15), y0 + rng.normal(0, 0.15)
        elif cls == 1:     # horizontal, bouncing at borders
            x = x0 + sign * speed * t
            x, y = np.clip(x, 3, S - 4), y0
        elif cls == 2:     # vertical
            y = np.clip(y0 + sign * speed * t, 3, S - 4)
            x = x0
        else:              # circular
            x = cx + r * np.cos(phase + omega * t)
            y = cy + r * np.sin(phase + omega * t)
        flicker = rng.normal(0, 0.01)
        frame = bg + (fg - bg) * _blob(x, y, S) + flicker
        frame += rng.normal(0, 0.03, size=(S, S))
        frames[t] = np.clip(frame, 0, 1)
    meta = dict(speed=float(speed), bg_level=float(bg), fg_level=float(fg))
    return frames, meta


def make_dataset(n, bias, seed):
    """Create n clips. bias=True -> spurious class/group correlation.

    Returns X (n,T,H,W) float32, and a Polars DataFrame of metadata.
    """
    rng = np.random.default_rng(seed)
    classes = np.tile(np.arange(4), n // 4)
    rng.shuffle(classes)
    X = np.empty((n, C.N_FRAMES, C.FRAME_SIZE, C.FRAME_SIZE), dtype=np.float32)
    rows = []
    for i, c in enumerate(classes):
        p_bright = 0.5
        if bias:
            p_bright = C.SPURIOUS_P if c in (0, 1) else 1 - C.SPURIOUS_P
        g = 0 if rng.random() < p_bright else 1
        X[i], meta = make_clip(int(c), g, rng)
        rows.append(dict(clip_idx=i, label=int(c), group=int(g), **meta))
    return X, pl.DataFrame(rows)


def make_stream(n_clips, seed):
    """Long 'video' made of consecutive random clips; returns frames and the
    ground-truth indices of scene cuts (first frame of every new clip)."""
    rng = np.random.default_rng(seed)
    clips, cuts = [], []
    for i in range(n_clips):
        c, g = int(rng.integers(0, 4)), int(rng.integers(0, 2))
        clip, _ = make_clip(c, g, rng)
        if i > 0:
            cuts.append(i * C.N_FRAMES)
        clips.append(clip)
    return np.concatenate(clips, axis=0), np.array(cuts)
