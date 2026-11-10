from typing import Optional

import torch
import torch.nn.functional as F


def cssl_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.1,
    hard_ratio: float = 1.0,
    alpha_max: float = 0.4,
    max_hard: int = 256,
    aux_negatives: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    memory=None,
):
    z = z.float()
    n = z.shape[0]
    dev = z.device
    pool, pool_lab = [z], [labels]
    if aux_negatives is not None and aux_negatives.numel():
        pool.append(aux_negatives.to(z))
        pool_lab.append(torch.full((aux_negatives.shape[0],), -1, device=dev, dtype=labels.dtype))
    if memory is not None and memory[0].numel():
        pool.append(memory[0].to(z).detach())
        pool_lab.append(memory[1].to(device=dev, dtype=labels.dtype))
    pool, pool_lab = torch.cat(pool, 0), torch.cat(pool_lab)
    sim = z @ pool.t()
    same = labels[:, None] == pool_lab[None, :]
    eye = torch.zeros_like(same)
    eye[:, :n] = torch.eye(n, dtype=torch.bool, device=dev)
    pos = same & ~eye
    neg = ~same
    n_pos, n_neg = pos.sum(1), neg.sum(1)
    valid = (n_pos > 0) & (n_neg > 0)
    if not bool(valid.any()):
        return z.sum() * 0.0, {"cl_anchors": 0, "cl_hard": 0}

    logits = sim / tau
    neg_logits = logits.masked_fill(~neg, float("-inf"))

    h = int(min(max_hard, max(1, round(hard_ratio * float(n_neg.max())))))
    if hard_ratio > 0:
        w = neg.float() + (~valid[:, None]).float() * 1e-6
        idx = torch.multinomial(w, h, replacement=True, generator=generator)
        alpha = torch.rand(n, h, 1, device=dev, generator=generator) * alpha_max
        zh = F.normalize(alpha * z.detach()[:, None, :] + (1 - alpha) * pool.detach()[idx], dim=-1)
        hard_logits = torch.einsum("nd,nhd->nh", z, zh) / tau
        keep = torch.arange(h, device=dev)[None, :] < (hard_ratio * n_neg.float()).clamp(max=h)[:, None]
        hard_logits = hard_logits.masked_fill(~keep, float("-inf"))
        neg_logits = torch.cat([neg_logits, hard_logits], 1)
    neg_lse = torch.logsumexp(neg_logits, 1)

    pair = -logits + torch.logaddexp(logits, neg_lse[:, None])
    per_anchor = (pair * pos).sum(1) / n_pos.clamp(min=1)
    loss = per_anchor[valid].mean()
    return loss, {"cl_anchors": int(valid.sum()), "cl_hard": h if hard_ratio > 0 else 0}


def gaussian_aux_negatives(dim: int, num: int, seed: int = 0, device="cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(dim, generator=g)
    a = torch.randn(dim, dim, generator=g) / dim ** 0.5
    x = mu + torch.randn(num, dim, generator=g) @ a.t()
    return F.normalize(x, dim=1).to(device)
