"""Unsupervised scene-cut detection on a long video stream.

Score = intensity-histogram distance + 3 x mean absolute difference of blurred frames
between consecutive frames. A cut is declared where the score exceeds
median + k * MAD (robust threshold). k is tuned on a validation stream.
"""
import cv2
import numpy as np


def cut_scores(frames, bins=16):
    h = np.stack([np.histogram(f, bins=bins, range=(0, 1))[0] / f.size for f in frames])
    hist_d = 0.5 * np.abs(np.diff(h, axis=0)).sum(1)          # total-variation distance
    blur = np.stack([cv2.GaussianBlur(f, (0, 0), 1.5) for f in frames])   # suppress sensor noise
    pix_d = np.abs(np.diff(blur, axis=0)).mean((1, 2))
    return hist_d + 3.0 * pix_d                                     # score[i]: frame i -> i+1


def detect(frames, k):
    s = cut_scores(frames)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-8
    return np.where(s > med + k * mad)[0] + 1                 # index of first frame of new shot


def prf(pred, truth, tol=1):
    truth = list(truth); matched = 0
    used = set()
    for p in pred:
        for j, t in enumerate(truth):
            if j not in used and abs(p - t) <= tol:
                used.add(j); matched += 1; break
    prec = matched / max(len(pred), 1)
    rec = matched / max(len(truth), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


def tune_k(frames, truth, grid=np.arange(2, 40, 1.0)):
    best = max(grid, key=lambda k: prf(detect(frames, k), truth)[2])
    return float(best)
