"""ViT backbone (timm, dynamic image size for 224 global / 96 local crops), DINO head, multi-crop wrapper.
Head and wrapper follow facebookresearch/dino vision_transformer.py / utils.py."""
import warnings

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


def build_backbone(arch="vit_small_patch16_224", drop_path=0.0):
    """CLS token after the final norm (num_classes=0 -> pooled 'token'). No pretrained weights (DINO from scratch)."""
    return timm.create_model(arch, pretrained=False, num_classes=0, drop_path_rate=drop_path, dynamic_img_size=True)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # torch.nn.utils.weight_norm is deprecated but matches DINO exactly
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        return self.last_layer(x)


class MultiCropWrapper(nn.Module):
    """Runs the backbone once per group of same-resolution crops, then the head on all features."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x, return_backbone=False):
        if not isinstance(x, (list, tuple)):
            x = [x]
        sizes = torch.tensor([inp.shape[-1] for inp in x])
        ends = torch.cumsum(torch.unique_consecutive(sizes, return_counts=True)[1], 0).tolist()
        feats, start = [], 0
        for end in ends:
            feats.append(self.backbone(torch.cat(x[start:end])))
            start = end
        feat = torch.cat(feats)
        out = self.head(feat)
        return (feat, out) if return_backbone else out
