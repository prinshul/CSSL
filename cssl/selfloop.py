import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def permutation_pool(k=100, n_tiles=9, seed=0):
    perms = np.array(list(itertools.permutations(range(n_tiles))), dtype=np.int8)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(perms)))]
    dist = (perms != perms[chosen[0]]).sum(1).astype(np.int32)
    taken = np.zeros(len(perms), bool)
    taken[chosen[0]] = True
    for _ in range(k - 1):
        j = int(np.argmax(np.where(taken, -1, dist)))
        chosen.append(j)
        taken[j] = True
        dist += (perms != perms[j]).sum(1)
    return perms[chosen].astype(np.int64)


class JigsawHead(nn.Module):

    def __init__(self, cin, k, hidden=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.mlp = nn.Sequential(nn.Linear(cin * 9, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, k))

    def forward(self, f):
        return self.mlp(self.pool(f.float()).flatten(1))


def _tiles(x, t):
    B, C = x.shape[:2]
    return x[:, :, :3 * t, :3 * t].reshape(B, C, 3, t, 3, t).permute(0, 2, 4, 1, 3, 5).reshape(B, 9, C, t, t)


def _untile(tiles, x, t):
    B, _, C = tiles.shape[:3]
    out = x.clone()
    out[:, :, :3 * t, :3 * t] = tiles.reshape(B, 3, 3, C, t, t).permute(0, 3, 1, 4, 2, 5).reshape(B, C, 3 * t, 3 * t)
    return out


def jigsaw(x, perm, rot):
    t = min(x.shape[-2:]) // 3
    tiles = _tiles(x, t)
    b = torch.arange(x.shape[0], device=x.device)[:, None]
    new = tiles[b, perm]
    for r in (1, 2, 3):
        m = rot == r
        if bool(m.any()):
            new[m] = torch.rot90(new[m], r, (2, 3))
    return _untile(new, x, t)


def unjigsaw(y, perm, rot):
    t = min(y.shape[-2:]) // 3
    tiles = _tiles(y, t).clone()
    for r in (1, 2, 3):
        m = rot == r
        if bool(m.any()):
            tiles[m] = torch.rot90(tiles[m], -r, (2, 3))
    orig = torch.empty_like(tiles)
    b = torch.arange(y.shape[0], device=y.device)[:, None]
    orig[b, perm] = tiles
    return _untile(orig, y, t)


def random_jigsaw_params(batch, pool, generator, device):
    idx = torch.randint(len(pool), (batch,), generator=generator)
    rot = torch.randint(4, (batch, 9), generator=generator)
    return idx.to(device), pool[idx].to(device), rot.to(device)


def jigsaw_loss(model, x, pool, q, generator, reduction="mean"):
    B = x.shape[0]
    idx, perm, rot = random_jigsaw_params(B * q, pool, generator, x.device)
    xs = jigsaw(x.repeat(q, 1, 1, 1), perm, rot)
    logits = model(xs, "jigsaw_only")
    return F.cross_entropy(logits.float(), idx, reduction=reduction), (logits.argmax(1) == idx).float().mean()


def self_loop_pseudo_label(model, x, pool, q, enc_optimizer, generator, amp_ctx):
    preds, losses = [], []
    for _ in range(q):
        idx, perm, rot = random_jigsaw_params(x.shape[0], pool, generator, x.device)
        with amp_ctx():
            logits, seg = model(jigsaw(x, perm, rot), "jigsaw")
            l = F.cross_entropy(logits.float(), idx, reduction="none")
        preds.append(unjigsaw(seg.detach().float().softmax(1), perm, rot))
        losses.append(l.detach())
        enc_optimizer.zero_grad(set_to_none=True)
        if bool(torch.isfinite(l).all()):
            l.mean().backward()
            enc_optimizer.step()
    L = torch.stack(losses)
    w = 1.0 - L / L.sum(0, keepdim=True).clamp(min=1e-12)
    w = w / w.sum(0, keepdim=True).clamp(min=1e-12)
    y_sl = (torch.stack(preds) * w[:, :, None, None, None]).sum(0)
    return y_sl, float(L.mean())


def uncertainty_guided_loss(pred_prob, y_sl, th, fg=1):
    m = (y_sl[:, fg] > th).float()
    se = ((pred_prob - y_sl) ** 2).sum(1)
    return (se * m).sum() / m.sum().clamp(min=1.0), float(m.mean())
