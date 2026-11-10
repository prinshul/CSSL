from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_class_patches(
    images: torch.Tensor,
    label_maps: torch.Tensor,
    classes: Sequence[int],
    patch: int = 64,
    stride: int = 32,
    min_pixels: int = 25,
    max_patches: int = 816,
    ignore_index: int = 255,
    generator: Optional[torch.Generator] = None,
    mode: str = "prominent",
) -> Dict[str, torch.Tensor]:
    B, C, H, W = images.shape
    dev = images.device
    s = patch
    if H < s or W < s:
        ph, pw = max(0, s - H), max(0, s - W)
        images = F.pad(images, (0, pw, 0, ph))
        label_maps = F.pad(label_maps, (0, pw, 0, ph), value=ignore_index)
        B, C, H, W = images.shape
    cls = torch.as_tensor(list(classes), device=dev)
    onehot = (label_maps[:, None] == cls[None, :, None, None]).float()
    counts = F.avg_pool2d(onehot, s, stride) * (s * s)
    if mode == "prominent":
        best_cnt, best_k = counts.max(1)
        b, iy, ix = (best_cnt >= min_pixels).nonzero(as_tuple=True)
        k = best_k[b, iy, ix]
    elif mode == "each":
        b, k, iy, ix = (counts >= min_pixels).nonzero(as_tuple=True)
    else:
        raise ValueError(mode)
    if b.numel() == 0:
        return {"patches": images.new_zeros(0, C, s, s), "masks": images.new_zeros(0, 1, s, s),
                "labels": torch.zeros(0, dtype=torch.long, device=dev),
                "boxes": torch.zeros(0, 3, dtype=torch.long, device=dev)}
    labels = cls[k]

    n = labels.numel()
    if n > max_patches:
        perm = torch.randperm(n, generator=generator).to(dev)
        lab = labels[perm]
        rank = torch.zeros(n, dtype=torch.long, device=dev)
        for c in lab.unique():
            sel = (lab == c).nonzero(as_tuple=True)[0]
            rank[sel] = torch.arange(sel.numel(), device=dev)
        keep = perm[torch.argsort(rank * (n + 1) + torch.arange(n, device=dev))[:max_patches]]
        b, iy, ix, labels = b[keep], iy[keep], ix[keep], labels[keep]

    y0, x0 = iy * stride, ix * stride
    ar = torch.arange(s, device=dev)
    rows = (y0[:, None] + ar[None, :])
    cols = (x0[:, None] + ar[None, :])
    crops = images[b[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    crops = crops.permute(0, 3, 1, 2)
    lab_crops = label_maps[b[:, None, None], rows[:, :, None], cols[:, None, :]]
    masks = (lab_crops == labels[:, None, None]).float()[:, None]
    return {"patches": crops * masks, "masks": masks, "labels": labels, "boxes": torch.stack([b, y0, x0], 1)}
