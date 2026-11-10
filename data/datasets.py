import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision import transforms

from .strong_aug import (HED_LIGHT, HED_STRONG, affine_resize, hed_jitter, mr_intensity, random_affine_params,
                         scaled_crop)
from .transform import blur, obtain_cutmix_box, random_rot_flip, random_rotate

MONUSEG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
MONUSEG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_split(split_dir, dataset, seed, prefix="seed"):
    return json.load(open(os.path.join(split_dir, dataset, "%s%d.json" % (prefix, seed))))


class Mixture(Dataset):

    def __init__(self, parts, n):
        self.parts = [(ds, float(sh)) for ds, sh in parts if sh > 0 and len(ds) > 0]
        cum = np.cumsum([sh for _, sh in self.parts])
        self.bounds = cum / cum[-1]
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = min(int(np.searchsorted(self.bounds, (i + 0.5) / self.n, side="right")), len(self.parts) - 1)
        ds = self.parts[k][0]
        return ds[random.randrange(len(ds))]


def _geom(img, mask):
    if random.random() > 0.5:
        return random_rot_flip(img, mask)
    if random.random() > 0.5:
        return random_rotate(img, mask)
    return img, mask


def _geom_side_preserving(img, mask):
    if random.random() > 0.5:
        return random_rotate(img, mask)
    return img, mask


def stack_context(vol, k, context):
    idx = np.clip(np.arange(k - context, k + context + 1), 0, vol.shape[0] - 1)
    return vol[idx]


class ChaosSlices(Dataset):
    def __init__(self, root, subjects, mode, size=256, nsample=None, geom="side_preserving", context=0,
                 aug="unimatch", affine=None):
        self.mode, self.size = mode, size
        self.geom = {"side_preserving": _geom_side_preserving, "unimatch": _geom}[geom]
        self.context, self.aug = int(context), aug
        if aug not in ("unimatch", "strong"):
            raise ValueError(aug)
        if aug == "unimatch" and self.context:
            raise ValueError("2.5D input (context > 0) needs aug=strong")
        self.affine = {**dict(rot=15.0, scale=(0.85, 1.2), shift=0.08, p=0.8), **(affine or {})}
        self.affine["scale"] = tuple(self.affine["scale"])
        self.vols = {s: np.load(os.path.join(root, "%s.npz" % s)) for s in subjects}
        self.vols = {s: (v["image"].astype(np.float32), v["label"].astype(np.uint8)) for s, v in self.vols.items()}
        self.items = [(s, k) for s in subjects for k in range(self.vols[s][0].shape[0])]
        if nsample is not None and mode == "train_l" and len(self.items) < nsample:
            reps = int(np.ceil(nsample / len(self.items)))
            self.items = (self.items * reps)[:nsample]

    def __len__(self):
        return len(self.items)

    def _strong_item(self, s, k):
        img = stack_context(self.vols[s][0], k, self.context)
        mask = self.vols[s][1][k]
        rot, scale, shift = random_affine_params(**self.affine)
        img, mask = affine_resize(img, mask, self.size, rot, scale, shift)
        if self.mode == "train_l":
            img = mr_intensity(img, "light")
            return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64))
        s1, s2 = mr_intensity(img, "strong"), mr_intensity(img, "strong")
        return (torch.from_numpy(img), torch.from_numpy(s1), torch.from_numpy(s2),
                obtain_cutmix_box(self.size, p=0.5), obtain_cutmix_box(self.size, p=0.5))

    def __getitem__(self, i):
        s, k = self.items[i]
        if self.aug == "strong":
            return self._strong_item(s, k)
        img, mask = self.vols[s][0][k], self.vols[s][1][k]
        img, mask = self.geom(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)
        if self.mode == "train_l":
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        weak = torch.from_numpy(np.array(pil)).unsqueeze(0).float() / 255.0
        strong = []
        for _ in range(2):
            t = deepcopy(pil)
            if random.random() < 0.8:
                t = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(t)
            t = blur(t, p=0.5)
            strong.append(torch.from_numpy(np.array(t)).unsqueeze(0).float() / 255.0)
        return weak, strong[0], strong[1], obtain_cutmix_box(self.size, p=0.5), obtain_cutmix_box(self.size, p=0.5)


class ChaosVolumes(Dataset):

    def __init__(self, root, subjects, context=0):
        self.root, self.subjects, self.context = root, subjects, int(context)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, i):
        v = np.load(os.path.join(self.root, "%s.npz" % self.subjects[i]))
        img = v["image"].astype(np.float32)
        if self.context:
            img = np.stack([stack_context(img, k, self.context) for k in range(img.shape[0])])
        else:
            img = img[:, None]
        return torch.from_numpy(img), torch.from_numpy(v["label"].astype(np.int64)), self.subjects[i]


def _norm_rgb(img_uint8):
    x = (img_uint8.astype(np.float32) / 255.0 - MONUSEG_MEAN) / MONUSEG_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class MonusegCrops(Dataset):

    def __init__(self, root, names, mode, size=256, crops_per_image=64, strong_aug="unimatch",
                 scale=(1.0, 1.0), labeled_color="none"):
        self.mode, self.size = mode, size
        self.strong_aug = strong_aug
        if strong_aug not in ("unimatch", "he", "hed") or labeled_color not in ("none", "hed"):
            raise ValueError((strong_aug, labeled_color))
        self.scale, self.labeled_color = tuple(scale), labeled_color
        split = "train"
        self.imgs = [np.array(Image.open(os.path.join(root, split, "images", n + ".png")).convert("RGB")) for n in names]
        self.masks = [np.array(Image.open(os.path.join(root, split, "masks", n + ".png"))) for n in names]
        self.n = len(names) * crops_per_image

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = i % len(self.imgs)
        img, mask = self.imgs[j], self.masks[j]
        h, w = mask.shape
        if h < self.size or w < self.size:
            ph, pw = max(0, self.size - h), max(0, self.size - w)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
            h, w = mask.shape
        if self.scale == (1.0, 1.0):
            y, x = random.randint(0, h - self.size), random.randint(0, w - self.size)
            img, mask = img[y:y + self.size, x:x + self.size], mask[y:y + self.size, x:x + self.size]
        else:
            img, mask = scaled_crop(img, mask, self.size, self.scale)
        if random.random() > 0.5:
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if random.random() > 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        if self.mode == "train_l":
            if self.labeled_color == "hed" and random.random() < 0.8:
                img = hed_jitter(img, *HED_LIGHT)
            return _norm_rgb(img), torch.from_numpy(mask.astype(np.int64))
        pil = Image.fromarray(img)
        strong = []
        for _ in range(2):
            t = deepcopy(pil)
            if self.strong_aug == "hed":
                if random.random() < 0.8:
                    t = Image.fromarray(hed_jitter(np.array(t), *HED_STRONG))
                if random.random() < 0.5:
                    t = transforms.ColorJitter(0.2, 0.2, 0.2, 0.04)(t)
            elif self.strong_aug == "he":
                if random.random() < 0.8:
                    t = transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)(t)
            else:
                if random.random() < 0.8:
                    t = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(t)
                t = transforms.RandomGrayscale(p=0.2)(t)
            t = blur(t, p=0.5)
            strong.append(_norm_rgb(np.array(t)))
        return _norm_rgb(img), strong[0], strong[1], obtain_cutmix_box(self.size, p=0.5), obtain_cutmix_box(self.size, p=0.5)


class MonusegTest(Dataset):
    def __init__(self, root, names, split="test"):
        self.root, self.names, self.split = root, names, split

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.array(Image.open(os.path.join(self.root, self.split, "images", n + ".png")).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.root, self.split, "masks", n + ".png")))
        return _norm_rgb(img), torch.from_numpy(mask.astype(np.int64)), n
