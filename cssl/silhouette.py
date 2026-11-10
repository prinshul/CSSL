from typing import Optional

import torch


@torch.no_grad()
def silhouette_l1(z: torch.Tensor, labels: torch.Tensor, max_n: int = 1024,
                  generator: Optional[torch.Generator] = None) -> Optional[float]:
    if labels.unique().numel() < 2:
        return None
    if z.shape[0] > max_n:
        idx = torch.randperm(z.shape[0], generator=generator)[:max_n].to(z.device)
        z, labels = z[idx], labels[idx]
        if labels.unique().numel() < 2:
            return None
    z = z.float()
    d = torch.cdist(z, z, p=1)
    cls = labels.unique()
    onehot = (labels[:, None] == cls[None, :]).float()
    sizes = onehot.sum(0)
    sum_d = d @ onehot
    own = onehot.argmax(1)
    own_size = sizes[own]
    a = sum_d.gather(1, own[:, None]).squeeze(1) / (own_size - 1).clamp(min=1)
    mean_other = sum_d / sizes[None, :]
    mean_other = mean_other.masked_fill(onehot.bool(), float("inf"))
    b = mean_other.min(1).values
    s = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    s = torch.where(own_size > 1, s, torch.zeros_like(s))
    return float(s.mean())
