"""CIFAR-100 (or a synthetic stand-in for smoke tests) with DINO's multi-crop augmentation.
Augmentation = facebookresearch/dino main_dino.py DataAugmentationDINO (2 global + N local crops)."""
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import datasets, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # DINO uses ImageNet statistics
NUM_CLASSES = {"cifar100": 100, "fake": 100}


class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p, self.lo, self.hi = p, radius_min, radius_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.lo, self.hi)))


class Solarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


class DataAugmentationDINO:
    def __init__(self, global_scale=(0.4, 1.0), local_scale=(0.05, 0.4), local_crops=6, global_size=224,
                 local_size=96):
        flip_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        bic = transforms.InterpolationMode.BICUBIC
        self.g1 = transforms.Compose([transforms.RandomResizedCrop(global_size, scale=global_scale, interpolation=bic),
                                      flip_color, GaussianBlur(1.0), norm])
        self.g2 = transforms.Compose([transforms.RandomResizedCrop(global_size, scale=global_scale, interpolation=bic),
                                      flip_color, GaussianBlur(0.1), Solarization(0.2), norm])
        self.loc = transforms.Compose([transforms.RandomResizedCrop(local_size, scale=local_scale, interpolation=bic),
                                       flip_color, GaussianBlur(0.5), norm])
        self.local_crops = local_crops

    def __call__(self, img):
        return [self.g1(img), self.g2(img)] + [self.loc(img) for _ in range(self.local_crops)]


def eval_transform(size=224):
    return transforms.Compose([transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class FakeCIFAR(torch.utils.data.Dataset):
    """32x32 class-coloured noise, CIFAR-like API (smoke tests without downloads)."""

    def __init__(self, n, train, transform=None, classes=100):
        rng = np.random.default_rng(0 if train else 1)
        colors = np.random.default_rng(123).integers(0, 200, (classes, 3))
        self.targets = rng.integers(0, classes, n).tolist()
        self.data = np.clip(rng.integers(0, 56, (n, 32, 32, 3)) + colors[self.targets][:, None, None], 0, 255)
        self.data = self.data.astype(np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), self.targets[i]


def get_dataset(name, root, train, transform, limit=None, download=True):
    if name == "cifar100":
        ds = datasets.CIFAR100(root, train=train, transform=transform, download=download)
    elif name == "fake":
        ds = FakeCIFAR(limit or 512, train, transform)
    else:
        raise ValueError(name)
    if limit is not None and limit < len(ds):
        idx = np.random.default_rng(0).permutation(len(ds))[:limit].tolist()
        ds = torch.utils.data.Subset(ds, sorted(idx))
    return ds
