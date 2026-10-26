"""Hand-crafted video features: optical flow (Farneback) + frame statistics."""
import cv2
import numpy as np

FLOW_FEATURES = [
    "mean_abs_dx", "mean_abs_dy", "mean_mag", "std_dx_t", "std_dy_t",
    "frame_diff", "dxdy_corr",
] + [f"dir_hist_{i}" for i in range(8)]


def clip_features(clip):
    """clip: (T,H,W) float32 in [0,1] -> 1-D feature vector (15 values)."""
    g = (clip * 255).astype(np.uint8)
    dxs, dys, mags, hist = [], [], [], np.zeros(8)
    for t in range(len(g) - 1):
        f = cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, 0.5, 2, 7, 3, 5, 1.1, 0)
        fx, fy = f[..., 0], f[..., 1]
        mag = np.hypot(fx, fy)
        w = mag > np.percentile(mag, 95)          # focus on the moving blob
        dxs.append(fx[w].mean()); dys.append(fy[w].mean()); mags.append(mag[w].mean())
        ang = (np.arctan2(fy[w], fx[w]) + np.pi) / (2 * np.pi) * 8
        hist += np.bincount(np.clip(ang.astype(int), 0, 7), weights=mag[w], minlength=8)
    dxs, dys = np.array(dxs), np.array(dys)
    hist = hist / (hist.sum() + 1e-8)
    corr = np.corrcoef(dxs, dys)[0, 1] if dxs.std() > 1e-6 and dys.std() > 1e-6 else 0.0
    fd = np.abs(np.diff(clip, axis=0)).mean()
    return np.array([np.abs(dxs).mean(), np.abs(dys).mean(), np.mean(mags),
                     dxs.std(), dys.std(), fd, corr, *hist], dtype=np.float32)


def batch_features(X):
    return np.stack([clip_features(c) for c in X])
