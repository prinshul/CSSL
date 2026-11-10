#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from cssl import AdaptivePerturbation, sample_class_patches, silhouette_l1
from data.datasets import MonusegTest, load_split
from model.cssl_net import CSSLNet
from qualitative import best_window, dice, load_run

CLASSES = [1, 0, 2]
NAMES = {1: "nucleus", 0: "background", 2: "rim"}


def rim_labels(gt, w=2):
    near_bg = F.max_pool2d((gt == 0).float()[None, None], 2 * w + 1, 1, w)[0, 0] > 0
    return torch.where((gt == 1) & near_bg, torch.full_like(gt, 2), gt)


def pad32(x, lab=None):
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    if lab is not None:
        lab = F.pad(lab, (0, pw, 0, ph), value=255)
    return x, lab


@torch.no_grad()
def encode(net, x):
    if net.pm1:
        x = x * 2.0 - 1.0
    return net.unet.encoder(x)


@torch.no_grad()
def pooled_features(net, x, masks, boxes, level):
    f = encode(net, x)[level].float()
    stride = x.shape[-1] // f.shape[-1]
    s = masks.shape[-1]
    cs, ar = s // stride, torch.arange(s // stride, device=f.device)
    rows, cols = boxes[:, 1:2] // stride + ar, boxes[:, 2:3] // stride + ar
    crops = f[boxes[:, 0, None, None], :, rows[:, :, None], cols[:, None, :]]
    m = F.interpolate(masks.float(), size=(cs, cs), mode="area")[:, 0, :, :, None]
    den = m.sum((1, 2))
    p = (crops * m).sum((1, 2)) / den.clamp(min=1e-6)
    return torch.where(den > 1e-6, p, crops.mean((1, 2)))


def tsne(z, seed):
    from sklearn.manifold import TSNE
    y = TSNE(2, perplexity=40, init="pca", learning_rate="auto", random_state=seed).fit_transform(z)
    y = y - y.mean(0)
    return y / np.sqrt((y ** 2).sum(1).mean())


def align(ref, y):
    from scipy.linalg import orthogonal_procrustes
    r, _ = orthogonal_procrustes(y, ref)
    return y @ r


def sil_cos(z, lab):
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(z, lab, metric="cosine"))


def pca_rgb(feats, basis=None):
    c, h, w = feats.shape
    x = feats.reshape(c, -1).T.astype(np.float64)
    if basis is None:
        mu = x.mean(0)
        _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
        y = (x - mu) @ vt[:3].T
        lo, hi = np.percentile(y, 1, 0), np.percentile(y, 99, 0)
        basis = (mu, vt[:3], lo, hi)
    mu, v, lo, hi = basis
    y = ((x - mu) @ v.T - lo) / np.maximum(hi - lo, 1e-9)
    return (np.clip(y, 0, 1).reshape(h, w, 3) * 255).astype(np.uint8), basis


def curves(exp, tag, key, k):
    f = os.path.join(exp, tag, "%s_s%d" % (key, k), "metrics.jsonl")
    if not os.path.exists(f):
        return None
    rec = [r for r in map(json.loads, open(f)) if "omega" in r or "lam" in r]
    out = {}
    for name in ("iter", "omega", "lam", "loss_cl", "loss_cr", "w_cl"):
        v = [r.get(name) for r in rec]
        out[name] = [None if (x is None or not np.isfinite(x)) else float(x) for x in v]
    step = max(1, len(rec) // 600)
    return {k2: v[::step] for k2, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="video_assets")
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--image", default=None)
    ap.add_argument("--split-dir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = os.path.abspath(a.exp)

    runs = {"recipe": os.path.join(E, "mabl_none", "monuseg_50_s0"),
            "cssl": os.path.join(E, "cssl_strongest", "monuseg_50_s0")}
    nets = {}
    for k, d in runs.items():
        r = load_run(d, ["final.pth"], dev)
        if r is None:
            sys.exit("missing run: %s" % d)
        nets[k] = r
        print("loaded %-7s %s/final.pth (test nuclei F1 %.2f)" % (k, d, r[3].get("test_mean_dice", float("nan"))))
    net_c, cfg, _, res_c = nets["cssl"]
    level = int(cfg["cssl"]["cl"].get("embed_level", -3))
    torch.manual_seed(0)
    init = CSSLNet(cfg["in_chns"], cfg["nclass"], tuple(cfg["widths"]), cfg["cssl"]["cl"]["embed_dim"],
                   backbone=cfg.get("backbone", "unet"), pretrained=False, input_norm=cfg.get("input_norm", "auto"),
                   jigsaw_classes=0, embed_level=level).to(dev).eval()
    models = {"init": init, "recipe": nets["recipe"][0], "cssl": net_c}

    sd = cfg.get("split_dir") or "splits"
    cands = [a.split_dir] if a.split_dir else [sd if os.path.isabs(sd) else os.path.join(os.path.dirname(E), sd),
                                               os.path.join(REPO, sd)]
    sdir = next((c for c in cands if os.path.exists(os.path.join(c, "monuseg"))), None)
    if sdir is None:
        sys.exit("splits not found in %s; pass --split-dir <repo used for training>/splits" % " or ".join(cands))
    split = load_split(sdir, "monuseg", 0, cfg.get("split_prefix", "seed"))
    print("splits: %s" % sdir)
    test = MonusegTest(cfg["data_root"], split["test"])
    feats = {k: [] for k in models}
    proj, labels, where = [], [], []
    for i in range(len(test)):
        x, gt, name = test[i]
        lab = rim_labels(gt)
        xp, lp = pad32(x[None].to(dev), lab[None].to(dev))
        P = sample_class_patches(xp, lp, CLASSES, 64, 32, 25, 10 ** 9, mode="each")
        for k, net in models.items():
            feats[k].append(F.normalize(pooled_features(net, xp, P["masks"], P["boxes"], level), dim=1).cpu())
        with torch.no_grad():
            z = net_c(xp, "embed_image", masks=P["masks"], boxes=P["boxes"]).float().cpu()
        proj.append(z)
        labels.append(P["labels"].cpu())
        where += [[i, int(b[1]), int(b[2])] for b in P["boxes"].cpu()]
        print("  test image %2d/%d %s: %d class-wise patches" % (i + 1, len(test), name, P["labels"].numel()))
    labels = torch.cat(labels).numpy()
    g = np.random.default_rng(0)
    keep = np.concatenate([g.choice(np.where(labels == c)[0], min(a.per_class, int((labels == c).sum())), replace=False)
                           for c in CLASSES])
    keep.sort()
    lab = labels[keep]
    emb = {"classes": [NAMES[c] for c in CLASSES], "label": [CLASSES.index(int(c)) for c in lab],
           "patch": [where[j] for j in keep], "level": level, "test_names": list(split["test"]),
           "n_total": int(labels.size), "models": {}}
    ref = None
    spaces = [(k, torch.cat(feats[k])[keep].numpy()) for k in ("init", "recipe", "cssl")]
    spaces.append(("cssl_proj", torch.cat(proj)[keep].numpy()))
    gen = torch.Generator().manual_seed(0)
    for k, z in spaces:
        s_l1 = silhouette_l1(torch.from_numpy(z), torch.from_numpy(lab), max_n=len(lab), generator=gen)
        s_c = sil_cos(z, lab)
        y = tsne(z, 0)
        if k == "recipe":
            ref = y
        elif ref is not None:
            y = align(ref, y)
        emb["models"][k] = {"xy": np.round(y, 4).tolist(), "silhouette_l1": s_l1, "silhouette_cos": s_c}
        print("%-10s silhouette (L1, as in training) %.3f   cosine %.3f   (%d patches)" % (k, s_l1, s_c, len(lab)))
    json.dump(emb, open(os.path.join(a.out, "embed.json"), "w"))

    cur = {"monuseg_s%d" % k: curves(E, "cssl_strongest", "monuseg_50", k) for k in range(3)}
    cur.update({"chaos_s%d" % k: curves(E, "cssl_strongest", "chaos_25", k) for k in range(3)})
    cur = {k: v for k, v in cur.items() if v}
    json.dump(cur, open(os.path.join(a.out, "curves.json"), "w"))
    for k, v in cur.items():
        om = [x for x in v["omega"] if x is not None]
        lm = [x for x in v["lam"] if x is not None]
        print("curves %-11s %d points, omega %.3f -> %.3f, lambda %.3f -> %.3f" % (
            k, len(v["iter"]), om[0] if om else float("nan"), om[-1] if om else float("nan"),
            lm[0] if lm else float("nan"), lm[-1] if lm else float("nan")))

    qj = os.path.join(REPO, "paper", "figures", "fig_qual_monuseg.json")
    name = a.image or (json.load(open(qj))["images"][0] if os.path.exists(qj) else split["test"][0])
    i = list(split["test"]).index(name) if name in split["test"] else 0
    name = split["test"][i]
    x, gt, _ = test[i]
    rgb = np.array(Image.open(os.path.join(cfg["data_root"], "test", "images", name + ".png")).convert("RGB"))
    lab = rim_labels(gt).numpy().astype(np.uint8)
    Image.fromarray(rgb).save(os.path.join(a.out, "image.png"))
    Image.fromarray(lab).save(os.path.join(a.out, "labels.png"))
    y0, x0 = best_window(gt.numpy(), 512)
    ws = min(512, *gt.shape)
    xc = x[None, :, y0:y0 + ws, x0:x0 + ws].to(dev)
    ck = torch.load(os.path.join(runs["cssl"], "final.pth"), map_location="cpu", weights_only=False)
    omega = float(ck["perturb"]["omega"])
    pert = AdaptivePerturbation(cfg["cssl"]["cr"]["epsilon"], 1.0, cfg["cssl"]["cr"].get("relative", True),
                                init_omega=omega)
    with torch.no_grad():
        f = encode(net_c, xc)
        torch.manual_seed(0)
        fp = pert(f)
        p_clean = net_c.unet.decoder(f).argmax(1)[0].cpu().numpy()
        p_noisy = net_c.unet.decoder(fp).argmax(1)[0].cpu().numpy()
    gtc = gt.numpy()[y0:y0 + ws, x0:x0 + ws]
    for lv in (1, 2):
        img, basis = pca_rgb(f[lv][0].float().cpu().numpy())
        img2, _ = pca_rgb(fp[lv][0].float().cpu().numpy(), basis)
        Image.fromarray(img).save(os.path.join(a.out, "feat_l%d_clean.png" % lv))
        Image.fromarray(img2).save(os.path.join(a.out, "feat_l%d_noisy.png" % lv))
    Image.fromarray((p_clean * 255).astype(np.uint8)).save(os.path.join(a.out, "pred_clean.png"))
    Image.fromarray((p_noisy * 255).astype(np.uint8)).save(os.path.join(a.out, "pred_noisy.png"))
    meta = {"image": name, "crop": [int(y0), int(x0), int(ws)], "omega_final": omega, "lambda_final": pert.lam,
            "f1_clean": 100 * dice(p_clean, gtc, 1), "f1_noisy": 100 * dice(p_noisy, gtc, 1),
            "agreement": 100 * dice(p_noisy, p_clean, 1), "levels": {"1": "1/4 resolution", "2": "1/8 resolution"},
            "test_f1": {k: nets[k][3].get("test_mean_dice") for k in nets}}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("image %s crop %s: omega %.3f, lambda %.3f; nuclei F1 clean %.1f, with feature noise %.1f, agreement %.1f" % (
        name, meta["crop"], omega, pert.lam, meta["f1_clean"], meta["f1_noisy"], meta["agreement"]))
    print("wrote %s (send this folder)" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
