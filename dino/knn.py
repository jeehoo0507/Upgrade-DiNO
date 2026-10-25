"""Weighted k-NN evaluation on frozen features (facebookresearch/dino eval_knn.py) + collapse statistics."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract(backbone, loader, device, amp=True):
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            f = backbone(x.to(device, non_blocking=True))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_classifier(train_f, train_l, test_f, test_l, k=20, T=0.07, num_classes=100, chunk=500, device="cpu"):
    train_f, train_l = train_f.to(device), train_l.to(device)
    k = min(k, train_f.shape[0])
    top1 = top5 = 0
    for i in range(0, test_f.shape[0], chunk):
        f, y = test_f[i:i + chunk].to(device), test_l[i:i + chunk].to(device)
        sim, idx = (f @ train_f.T).topk(k, dim=1)
        cand = train_l[idx]
        w = (sim / T).exp()
        votes = torch.zeros(f.shape[0], num_classes, device=device).scatter_add_(1, cand, w)
        pred = votes.argsort(1, descending=True)
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred[:, :min(5, num_classes)] == y[:, None]).any(1).sum().item()
    n = test_f.shape[0]
    return 100.0 * top1 / n, 100.0 * top5 / n


@torch.no_grad()
def feature_stats(f):
    """f: L2-normalised features (N, D). feat_std = mean per-dimension std x sqrt(D) (~1 for spread-out features,
    -> 0 under collapse to a point); mean_cos = mean pairwise cosine (-> 1 under collapse)."""
    f = f.float()
    std = f.std(0).mean().item() * math.sqrt(f.shape[1])
    m = f.mean(0)
    n = f.shape[0]
    mean_cos = ((m @ m) * n * n - n) / max(1, n * (n - 1))
    return {"feat_std": std, "mean_cos": float(mean_cos)}
