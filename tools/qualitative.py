#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data.datasets import ChaosVolumes, MonusegTest, load_split
from model.cssl_net import CSSLNet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

INK, INK2 = "#0b0b0b", "#52514e"
TP, FP, FN = np.array([0.93, 0.63, 0.0]), np.array([0.89, 0.29, 0.28]), np.array([0.38, 0.31, 0.84])
ORGANS = ["liver", "right kidney", "left kidney", "spleen"]
ORGAN_RGB = np.array([[0.93, 0.63, 0.0], [0.91, 0.48, 0.64], [0.38, 0.31, 0.84], [0.0, 0.51, 0.0]])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})


def load_run(run_dir, ckpts, device):
    if not os.path.exists(os.path.join(run_dir, "config.yaml")):
        return None
    path = next((os.path.join(run_dir, c) for c in ckpts if os.path.exists(os.path.join(run_dir, c))), None)
    if path is None:
        return None
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    cl = cfg["cssl"]["cl"]
    net = CSSLNet(cfg["in_chns"], cfg["nclass"], tuple(cfg["widths"]), cl["embed_dim"],
                  backbone=cfg.get("backbone", "unet"), pretrained=False, input_norm=cfg.get("input_norm", "auto"),
                  jigsaw_classes=0, embed_level=int(cl.get("embed_level", -1)))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    use_ema = bool((cfg.get("ema") or {}).get("enable", False)) and "ema" in ck
    state = ck["ema"] if use_ema else ck["model"]
    state = {k: v for k, v in state.items() if not k.startswith("jigsaw.")}
    net.load_state_dict(state, strict=False)
    missing = [k for k in net.state_dict() if k not in state and not k.startswith(("proj", "jigsaw"))]
    if missing:
        raise RuntimeError("%s: weights missing for %s" % (path, missing[:5]))
    res_f = os.path.join(run_dir, "results.json")
    res = json.load(open(res_f)) if os.path.exists(res_f) else {}
    return net.to(device).eval(), cfg, os.path.basename(path), res


def dice(p, g, c):
    a, b = p == c, g == c
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int((a & b).sum()) / s


@torch.no_grad()
def predict_monuseg(net, x, device):
    x = x[None].to(device)
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
    return net(x).float()[..., :h, :w].argmax(1)[0].cpu().numpy()


@torch.no_grad()
def predict_chaos(net, vol, crop, device):
    h, w = vol.shape[-2:]
    out = []
    for k in range(0, vol.shape[0], 16):
        v = F.interpolate(vol[k:k + 16].to(device), (crop, crop), mode="bilinear", align_corners=False)
        o = F.interpolate(net(v).float(), (h, w), mode="bilinear", align_corners=False)
        out.append(o.argmax(1).cpu())
    return torch.cat(out).numpy()


def dafnet_dice(pred, gt, nclass):
    P = np.stack([pred == c for c in range(1, nclass)], -1).astype(np.float64)
    G = np.stack([gt == c for c in range(1, nclass)], -1).astype(np.float64)
    inter, den = (P * G).sum((1, 2, 3)), P.sum((1, 2, 3)) + G.sum((1, 2, 3))
    return float(((2 * inter + 1e-12) / (den + 1e-12)).mean())


def check(name, got, res, key="test_mean_dice"):
    want = res.get(key)
    if want is None:
        print("  %-26s re-scored %.2f (no results.json value to compare)" % (name, got))
        return
    ok = abs(got - want) < 0.05
    print("  %-26s re-scored %.2f, results.json %.2f  %s" % (name, got, want, "OK" if ok else "MISMATCH"))
    if not ok:
        print("    (the figure would not show the reported model; check the run directory)")


def error_overlay(rgb, pred, gt, alpha=0.62):
    out = rgb.astype(np.float64) / 255.0
    for m, col in (((pred == 1) & (gt == 1), TP), ((pred == 1) & (gt == 0), FP), ((pred == 0) & (gt == 1), FN)):
        out[m] = (1 - alpha) * out[m] + alpha * col
    return out


def label_overlay(gray, lab, alpha=0.55):
    g = (gray - gray.min()) / max(1e-6, float(gray.max() - gray.min()))
    out = np.repeat(g[..., None], 3, -1)
    for c in range(1, len(ORGANS) + 1):
        m = lab == c
        out[m] = (1 - alpha) * out[m] + alpha * ORGAN_RGB[c - 1]
    return out


def best_window(gt, size=256):
    h, w = gt.shape
    size = min(size, h, w)
    ii = np.pad(np.cumsum(np.cumsum(gt == 1, 0), 1), ((1, 0), (1, 0)))
    best, arg = -1, (0, 0)
    for y in range(0, max(1, h - size + 1), 16):
        for x in range(0, max(1, w - size + 1), 16):
            s = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if s > best:
                best, arg = s, (y, x)
    return arg


def tidy(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def monuseg(args, device):
    E = args.exp
    runs = [("Supervised", os.path.join(E, args.sup_tag, "monuseg_50_s0")),
            ("Recipe only", os.path.join(E, "mabl_none", "monuseg_50_s0")),
            ("CSSL (ours)", os.path.join(E, "cssl_strongest", "monuseg_50_s0"))]
    nets = []
    for name, d in runs:
        r = load_run(d, ["final.pth", "latest.pth"], device)
        if r is None:
            print("  %-26s not found (%s): column skipped" % (name, d))
            continue
        nets.append((name,) + r)
    names = [n[0] for n in nets]
    if "Recipe only" not in names or "CSSL (ours)" not in names:
        print("MoNuSeg: recipe or CSSL run missing, figure skipped")
        return
    cfg = nets[-1][2]
    split = load_split(args.split_dir or os.path.join(REPO, cfg.get("split_dir") or "splits"), "monuseg", 0,
                       cfg.get("split_prefix", "seed"))
    test = MonusegTest(cfg["data_root"], split["test"])
    preds = {n[0]: [] for n in nets}
    gts, rgbs = [], []
    from PIL import Image
    for i in range(len(test)):
        x, gt, name = test[i]
        gts.append(gt.numpy())
        rgbs.append(np.array(Image.open(os.path.join(cfg["data_root"], "test", "images", name + ".png")).convert("RGB")))
        for n, net, *_ in nets:
            preds[n].append(predict_monuseg(net, x, device))
    print("MoNuSeg test set, nuclei F1 (per image, averaged):")
    per = {}
    for n, net, c, ck, res in nets:
        per[n] = np.array([dice(p, g, 1) for p, g in zip(preds[n], gts)]) * 100
        check("%s (%s)" % (n, ck), float(per[n].mean()), res)
    gain = per["CSSL (ours)"] - per["Recipe only"]
    order = np.argsort(-gain)
    picks = [("largest gain", order[0]), ("median gain", order[len(order) // 2]), ("smallest gain", order[-1])]
    cols = ["Image", "Ground truth"] + names
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(1.42 * len(cols), 1.5 * len(picks) + 0.25),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for r, (why, i) in enumerate(picks):
        y, x = best_window(gts[i])
        ws = min(256, *gts[i].shape)
        sl = (slice(y, y + ws), slice(x, x + ws))
        rgb, gt = rgbs[i][sl], gts[i][sl]
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(error_overlay(rgb, gt, gt))
        for c, n in enumerate(names):
            p = preds[n][i][sl]
            axes[r, 2 + c].imshow(error_overlay(rgb, p, gt))
            axes[r, 2 + c].text(4, ws - 6, "%.1f" % per[n][i], color="white", fontsize=7.4, va="bottom", ha="left",
                                fontweight="bold" if n == "CSSL (ours)" else "normal",
                                bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.55), lw=0))
        axes[r, 0].set_ylabel("%s\n(%+.1f)" % (why, gain[i]), fontsize=7.2, color=INK2)
        for a in axes[r]:
            tidy(a)
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=7.8, color=INK, fontweight="bold" if t == "CSSL (ours)" else "normal")
    fig.subplots_adjust(bottom=0.07, top=0.93)
    fig.legend([Patch(color=TP), Patch(color=FP), Patch(color=FN)],
               ["nucleus, correct", "false positive", "missed nucleus"], loc="upper center", ncol=3,
               fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, 0.06), handlelength=1.2)
    save(fig, args.out, "fig_qual_monuseg")
    json.dump({"images": [split["test"][i] for _, i in picks], "rules": [w for w, _ in picks],
               "per_image_nuclei_f1": {n: per[n].tolist() for n in per}, "test_names": list(split["test"])},
              open(os.path.join(args.out, "fig_qual_monuseg.json"), "w"), indent=1)


def chaos(args, device):
    E = args.exp
    runs = [("Supervised", args.sup_tag), ("Recipe only", "cabl_none"), ("CSSL (ours)", "cssl_strongest")]
    rows = []
    print("CHAOS test sets, DAFNet's Dice (best-validation checkpoint):")
    for k in range(3):
        nets = []
        for name, tag in runs:
            r = load_run(os.path.join(E, tag, "chaos_25_s%d" % k), ["best.pth", "final.pth"], device)
            if r is None:
                print("  split %d %-18s not found: skipped" % (k, name))
                continue
            nets.append((name,) + r)
        if not any(n[0] == "CSSL (ours)" for n in nets):
            continue
        cfg = nets[-1][2]
        split = load_split(args.split_dir or os.path.join(REPO, cfg.get("split_dir") or "splits"), "chaos", k,
                           cfg.get("split_prefix", "dafnet"))
        ds = ChaosVolumes(cfg["data_root"], ["%s_t2" % s for s in split["test"]], int(cfg.get("chaos_context", 0)))
        best, scores = None, {n[0]: [] for n in nets}
        for v in range(len(ds)):
            vol, gt, sid = ds[v]
            gt = gt.numpy()
            pv = {n: predict_chaos(net, vol, cfg["crop_size"], device) for n, net, *_ in nets}
            for n in pv:
                scores[n].append(dafnet_dice(pv[n], gt, cfg["nclass"]))
            area = (gt > 0).sum((1, 2))
            s = int(np.argmax(area))
            if best is None or area[s] > best[0]:
                ctx = int(cfg.get("chaos_context", 0))
                best = (area[s], sid, s, vol[s, ctx].numpy(), gt[s], {n: pv[n][s] for n in pv})
        for n, net, c, ck, res in nets:
            check("split %d %s (%s)" % (k, n, ck), float(np.mean(scores[n]) * 100), res)
        rows.append((k,) + best[1:])
    if not rows:
        print("CHAOS: no CSSL runs found, figure skipped")
        return
    names = [n for n, _ in runs if all(n in r[5] for r in rows)]
    cols = ["T2 image", "Ground truth"] + names
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.42 * len(cols), 1.5 * len(rows) + 0.25),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05}, squeeze=False)
    for r, (k, sid, s, img, gt, pv) in enumerate(rows):
        g = (img - img.min()) / max(1e-6, float(img.max() - img.min()))
        axes[r, 0].imshow(g, cmap="gray")
        axes[r, 1].imshow(label_overlay(img, gt))
        for c, n in enumerate(names):
            axes[r, 2 + c].imshow(label_overlay(img, pv[n]))
            d = np.mean([dice(pv[n], gt, o) for o in range(1, len(ORGANS) + 1) if (gt == o).any() or (pv[n] == o).any()])
            axes[r, 2 + c].text(4, img.shape[0] - 6, "%.1f" % (100 * d), color="white", fontsize=7.4, va="bottom",
                                fontweight="bold" if n == "CSSL (ours)" else "normal",
                                bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.55), lw=0))
        axes[r, 0].set_ylabel("split %d" % k, fontsize=7.4, color=INK2)
        for a in axes[r]:
            tidy(a)
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=7.8, color=INK, fontweight="bold" if t == "CSSL (ours)" else "normal")
    fig.subplots_adjust(bottom=0.07, top=0.93)
    fig.legend([Patch(color=c) for c in ORGAN_RGB], ORGANS, loc="upper center", ncol=4, fontsize=7.2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), handlelength=1.2)
    save(fig, args.out, "fig_qual_chaos")
    json.dump({"slices": [{"split": k, "subject": sid, "slice": int(s)} for k, sid, s, *_ in rows],
               "rule": "largest ground-truth organ area among the split's test slices"},
              open(os.path.join(args.out, "fig_qual_chaos.json"), "w"), indent=1)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=os.path.join(REPO, "exp"), help="experiment root (exp/)")
    ap.add_argument("--out", default=os.path.join(REPO, "figures"))
    ap.add_argument("--sup-tag", default="main_sup", help="tag of the supervised runs (column skipped if absent)")
    ap.add_argument("--split-dir", default=None, help="default: the run's split_dir")
    ap.add_argument("--only", choices=["monuseg", "chaos"], default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.only in (None, "monuseg"):
        monuseg(args, device)
    if args.only in (None, "chaos"):
        chaos(args, device)


if __name__ == "__main__":
    main()
