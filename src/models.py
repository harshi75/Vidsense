"""PyTorch model and loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tiny3DCNN(nn.Module):
    """Small spatio-temporal CNN: input (B,1,T,H,W) -> logits (B,n_classes)."""
    def __init__(self, n_classes=4, emb=32):
        super().__init__()
        def block(i, o, pool):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool3d(pool))
        self.features = nn.Sequential(block(1, 8, (1, 2, 2)), block(8, 16, (2, 2, 2)),
                                      block(16, emb, (2, 2, 2)), nn.AdaptiveAvgPool3d(1))
        self.head = nn.Linear(emb, n_classes)

    def embed(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


def focal_loss(logits, target, gamma=2.0, weight=None, reduction="none"):
    """Focal loss FL = -(1-p_t)^gamma * log(p_t); down-weights easy examples."""
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss if weight is None else loss * weight


def loss_fn(name):
    """Return f(logits, target, sample_weight) -> scalar."""
    def ce(l, y, w):
        v = F.cross_entropy(l, y, reduction="none")
        return (v * w).sum() / w.sum()
    def focal(l, y, w):
        v = focal_loss(l, y)
        return (v * w).sum() / w.sum()
    return {"ce": ce, "focal": focal}[name]
