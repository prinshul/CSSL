#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import textwrap
import wave

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from scipy.ndimage import binary_erosion, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(os.path.dirname(HERE), "numbers.json")))
W, H, FPS = 1920, 1080, 30

BG, SURF, SURF2, LINE = "#121211", "#1d1d1b", "#272725", "#3b3b38"
INK, INK2, MUTED = "#f0efec", "#bdbcb4", "#86857e"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#e3a20f", "#9085e9", "#e66767"
SUPC = "#1baf7a"
C_NUC, C_RIM, C_BG = VIOLET, ORANGE, AQUA
plt.rcParams.update({"font.family": "Inter", "mathtext.fontset": "cm", "text.color": INK})


def mean(v):
    return float(np.mean(v))


M, C = N["monuseg"], N["chaos"]
NUM = {
    "m_cssl": M["cssl_full"], "m_rec": M["recipe"], "m_sup": M["supervised_50_s0"]["nuclei_f1"],
    "m_sup100": M["supervised_100_s0"]["nuclei_f1"], "sl": M["selfloop_published"],
    "sl_sup": M["selfloop_published_sup50"], "daf": C["dafnet_published"],
    "c_cssl": C["cssl_full"]["best_val"], "c_rec": C["recipe"]["best_val"], "c_sup": C["supervised"]["best_val"],
}
m_gain = mean(NUM["m_cssl"]) - mean(NUM["m_rec"])
m_wins = sum(a > b for a, b in zip(NUM["m_cssl"], NUM["m_rec"]))
c_gain_b = mean(NUM["c_cssl"]) - mean(NUM["c_rec"])


def sgn(v):
    return ("+%.2f" % v) if v >= 0 else ("−%.2f" % -v)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def txt(ax, x, y, s, size=24, color=INK, a=1.0, ha="left", va="baseline", weight="regular", z=6, **kw):
    if a <= 0.003 or not s:
        return None
    return ax.text(x, y, s, fontsize=size, color=color, alpha=min(1, a), ha=ha, va=va, fontweight=weight,
                   zorder=z, **kw)


def rbox(ax, x, y, w, h, fc=SURF, ec=LINE, lw=1.6, a=1.0, r=16, z=2):
    if a <= 0.003:
        return
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec,
                                lw=lw, alpha=min(1, a), zorder=z))


def arrow(ax, p, q, color=INK2, lw=2.4, a=1.0, rad=0.0, z=4, ms=24):
    if a <= 0.003:
        return
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, alpha=min(1, a),
                                 connectionstyle="arc3,rad=%g" % rad, zorder=z, shrinkA=0, shrinkB=0))


def image(ax, img, x, y, w, h, a=1.0, z=3, interp="nearest", border=None):
    if a <= 0.003:
        return
    ax.imshow(img, extent=(x, x + w, y, y + h), alpha=min(1, a), zorder=z, interpolation=interp, aspect="auto")
    if border:
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=border, lw=1.5, alpha=min(1, a), zorder=z + 0.1))


def header(ax, kicker, title, a=1.0):
    txt(ax, 110, 975, kicker.upper(), 17, BLUE, a, weight="semibold")
    txt(ax, 110, 912, title, 42, INK, a, weight="bold")


def caption(ax, s, a=1.0):
    if not s:
        return
    rbox(ax, 110, 38, W - 220, 92, fc=SURF, ec=SURF, a=0.85 * a, r=18, z=8)
    txt(ax, W / 2, 84, textwrap.fill(s, 112), 22 if len(s) <= 112 else 20, INK2, a, ha="center", va="center", z=9,
        linespacing=1.4)


def tick(ax, x, y, s=26, color=BLUE, a=1.0):
    if a <= 0.003:
        return
    ax.add_patch(Circle((x, y), s, fc=color, ec="none", alpha=a, zorder=6))
    ax.plot([x - s * 0.45, x - s * 0.1, x + s * 0.5], [y, y - s * 0.38, y + s * 0.4], color=BG, lw=4.5,
            alpha=a, zorder=7, solid_capstyle="round", solid_joinstyle="round")


def tissue(n=256, k=40, seed=0, rmin=6.0, rmax=11.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    mask = np.zeros((n, n), bool)
    centers = []
    for _ in range(k):
        for _ in range(40):
            cy, cx = rng.uniform(rmax, n - rmax, 2)
            if all((cy - a) ** 2 + (cx - b) ** 2 > (1.7 * rmin) ** 2 for a, b in centers):
                break
        centers.append((cy, cx))
        ry, rx = rng.uniform(rmin, rmax, 2)
        th = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = (dx * np.cos(th) + dy * np.sin(th)) / rx
        v = (-dx * np.sin(th) + dy * np.cos(th)) / ry
        mask |= u * u + v * v <= 1
    tex = gaussian_filter(rng.normal(0, 1, (n, n)), 3.0)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    fib = gaussian_filter(rng.normal(0, 1, (n, n)), (0.8, 5.0))
    pink, pink2 = np.array([0.95, 0.73, 0.82]), np.array([0.84, 0.52, 0.68])
    img = pink * (1 - 0.45 * tex[..., None]) + pink2 * 0.45 * tex[..., None] + 0.05 * fib[..., None]
    chrom = gaussian_filter(rng.normal(0, 1, (n, n)), 0.9) > 0.25
    purple, dark = np.array([0.40, 0.23, 0.57]), np.array([0.20, 0.09, 0.36])
    nuc = np.where(chrom[..., None], dark, purple)
    soft = gaussian_filter(mask.astype(float), 0.7)[..., None]
    img = img * (1 - soft) + nuc * soft + rng.normal(0, 0.018, (n, n, 1))
    return np.clip(img, 0, 1), mask


def rims(mask, w=2):
    inner = binary_erosion(mask, iterations=w)
    return mask & ~inner, inner


def tint(img, m, col, a=0.6):
    out = img.copy()
    c = np.array(matplotlib.colors.to_rgb(col))
    out[m] = (1 - a) * out[m] + a * c
    return out


def silhouette_l1(z, lab):
    d = np.abs(z[:, None, :] - z[None, :, :]).sum(-1)
    s = []
    for i in range(len(z)):
        same = lab == lab[i]
        a_ = d[i, same].sum() / max(1, same.sum() - 1)
        b_ = min(d[i, lab == k].mean() for k in np.unique(lab) if k != lab[i])
        s.append((b_ - a_) / max(a_, b_, 1e-9))
    return float(np.mean(s))


def strong_view(img, seed):
    rng = np.random.default_rng(seed)
    out = np.clip(img, 1e-4, 1) ** rng.uniform(0.7, 1.4)
    gray = out.mean(-1, keepdims=True)
    out = gray + rng.uniform(0.6, 1.5) * (out - gray)
    out = out * rng.uniform(0.85, 1.1, 3)
    return np.clip(gaussian_filter(out, (0.6, 0.6, 0)), 0, 1)


def mask_img(m, col, bg=(0.11, 0.11, 0.10)):
    out = np.zeros(m.shape + (3,)) + np.array(bg)
    out[m] = np.array(matplotlib.colors.to_rgb(col))
    return out


TIS, TMASK = tissue(256, 46, seed=7)
TRIM, TINNER = rims(TMASK)
THUMBS = [tissue(96, 9, seed=100 + i, rmin=5, rmax=9) for i in range(30)]
LAB_ORDER = [int(i) for i in np.random.default_rng(5).permutation(30)[:15]]
WINS = [(y, x) for y in range(0, 193, 32) for x in range(0, 193, 32)]


def s_title(ax, t, D):
    a = 0.30 + 0.25 * seg(t, 0, 3)
    w = 1150 * (1 + 0.05 * t / D)
    image(ax, TIS, 1000, (H - w) / 2, w, w, a=a, z=1, interp="bilinear")
    grad = np.zeros((1, 256, 4))
    grad[..., :3] = matplotlib.colors.to_rgb(BG)
    grad[..., 3] = np.linspace(1, 0, 256) ** 1.4
    ax.imshow(grad, extent=(1000, 1500, 0, H), zorder=2, aspect="auto")
    ax.add_patch(Rectangle((0, 0), 1001, H, fc=BG, ec="none", zorder=1.5))
    k = seg(t, 0.3, 1.4)
    txt(ax, 120, 560 + 30 * (1 - k), "CSSL", 150, INK, k, weight="bold")
    k = seg(t, 1.0, 2.0)
    txt(ax, 126, 480, "Contrastive semi-supervised segmentation", 38, INK, k, weight="semibold")
    txt(ax, 126, 428, "with class-wise patches and", 30, INK2, k)
    txt(ax, 126, 386, "Silhouette-adaptive consistency", 30, INK2, k)
    chips = [("MoNuSeg 50%", "%.2f nuclei F1" % mean(NUM["m_cssl"])), ("CHAOS 25%", "%.2f Dice" % mean(NUM["c_cssl"])),
             ("Baselines", "Self-Loop, DAFNet")]
    x = 126
    for i, ((k1, k2), cw) in enumerate(zip(chips, (272, 236, 330))):
        k = seg(t, 2.2 + 0.35 * i, 2.9 + 0.35 * i)
        rbox(ax, x, 250, cw, 84, fc=SURF, ec=BLUE, lw=1.8, a=k, r=20, z=3)
        txt(ax, x + 22, 300, k1, 17, BLUE, k, weight="semibold")
        txt(ax, x + 22, 267, k2, 23, INK, k, weight="semibold")
        x += cw + 20
    txt(ax, 126, 170, "Method overview", 22, MUTED, seg(t, 3.6, 4.4))


def s_problem(ax, t, D):
    header(ax, "The problem", "Pixel labels are slow and expensive", seg(t, 0, 0.6))
    tw, gap = 128, 18
    x0 = (W - (10 * tw + 9 * gap)) / 2
    dim = seg(t, 6.2, 7.0)
    for i, (img, m) in enumerate(THUMBS):
        r, c = divmod(i, 10)
        x, y = x0 + c * (tw + gap), 700 - r * (tw + gap)
        k = seg(t, 0.3 + 0.03 * i, 0.9 + 0.03 * i)
        if i in LAB_ORDER:
            j = LAB_ORDER.index(i)
            kl = seg(t, 2.0 + 0.27 * j, 2.5 + 0.27 * j)
            rr, _ = rims(m, 1)
            image(ax, tint(img, rr, YELLOW, 0.95 * kl), x, y, tw, tw, a=k, border=YELLOW if kl > 0.5 else None)
        else:
            image(ax, img * (1 - 0.45 * dim), x, y, tw, tw, a=k)
    kl = seg(t, 6.0, 6.8)
    txt(ax, x0, 370, "15 labelled", 34, YELLOW, kl, weight="bold")
    txt(ax, x0 + 250, 370, "+  15 unlabelled images", 34, INK2, kl, weight="semibold")
    txt(ax, x0, 325, "MoNuSeg: 30 training images (1000 × 1000 px); in the labelled half every nucleus is outlined by hand",
        20, MUTED, kl)
    txt(ax, W - x0, 325, "illustration", 15, MUTED, kl, ha="right")
    caption(ax, "Semi-supervised segmentation learns from the labelled half and from the unlabelled half.",
            seg(t, 1.0, 1.6))


def s_recipe(ax, t, D):
    header(ax, "Step 0 · the recipe", "Weak-to-strong pseudo-labelling, identical in every run", seg(t, 0, 0.6))
    img, m = THUMBS[3]
    S = 210
    k0 = seg(t, 0.4, 1.0)
    image(ax, img, 110, 420, S, S, a=k0, border=LINE)
    txt(ax, 110 + S / 2, 390, "unlabelled image", 18, INK2, k0, ha="center")
    k1 = seg(t, 1.3, 1.9)
    arrow(ax, (330, 560), (440, 640), a=k1)
    image(ax, img, 450, 590, 170, 170, a=k1, border=LINE)
    txt(ax, 535, 565, "weak view", 18, INK2, k1, ha="center")
    arrow(ax, (630, 675), (700, 675), a=k1)
    rbox(ax, 710, 630, 210, 90, fc=SURF2, ec=LINE, a=k1)
    txt(ax, 815, 668, "EMA teacher", 22, INK, k1, ha="center", va="center", weight="semibold")
    k2 = seg(t, 2.6, 3.2)
    arrow(ax, (925, 675), (995, 675), a=k2)
    image(ax, mask_img(m, C_NUC), 1005, 590, 170, 170, a=k2, border=LINE)
    txt(ax, 1090, 565, "pseudo-label", 18, INK2, k2, ha="center")
    txt(ax, 1090, 540, "confidence ≥ 0.95", 16, MUTED, k2, ha="center")
    k3 = seg(t, 3.9, 4.5)
    arrow(ax, (330, 500), (440, 330), a=k3)
    cut = strong_view(img, 1).copy()
    cut[20:70, 40:96] = strong_view(THUMBS[11][0], 2)[20:70, 40:96]
    image(ax, strong_view(img, 3), 470, 210, 170, 170, a=k3 * 0.6, border=LINE)
    image(ax, cut, 450, 190, 170, 170, a=k3, border=LINE)
    if k3 > 0.01:
        sc = 170 / 96
        ax.add_patch(Rectangle((450 + 40 * sc, 190 + 170 - 70 * sc), 56 * sc, 50 * sc, fc="none", ec=YELLOW, lw=2,
                               ls="--", alpha=k3, zorder=5))
    txt(ax, 545, 160, "2 strong views + CutMix", 18, INK2, k3, ha="center")
    k4 = seg(t, 5.2, 5.8)
    arrow(ax, (650, 290), (700, 290), a=k4)
    rbox(ax, 710, 245, 210, 90, fc=SURF2, ec=LINE, a=k4)
    txt(ax, 815, 283, "student", 22, INK, k4, ha="center", va="center", weight="semibold")
    arrow(ax, (925, 290), (995, 290), a=k4)
    image(ax, mask_img(m, C_NUC), 1005, 205, 170, 170, a=k4, border=LINE)
    txt(ax, 1090, 180, "prediction", 18, INK2, k4, ha="center")
    k5 = seg(t, 6.4, 7.0)
    arrow(ax, (1090, 530), (1090, 385), color=BLUE, a=k5, lw=2.8)
    txt(ax, 1070, 478, r"$\mathcal{L}_{s}$: Dice", 22, INK, k5, ha="right", va="center")
    txt(ax, 1070, 444, "on confident pixels", 18, INK2, k5, ha="right", va="center")
    k6 = seg(t, 7.6, 8.3)
    rbox(ax, 1450, 300, 360, 430, fc=SURF, ec=LINE, a=k6)
    lim, lm = THUMBS[0]
    image(ax, lim, 1480, 520, 140, 140, a=k6, border=LINE)
    image(ax, mask_img(lm, C_NUC), 1640, 520, 140, 140, a=k6, border=LINE)
    txt(ax, 1630, 680, "labelled image + mask", 18, INK2, k6, ha="center")
    txt(ax, 1630, 450, r"$\mathcal{L}_{sup}=\frac{1}{2}(\mathrm{CE}+\mathrm{Dice})$", 26, INK, k6, ha="center")
    txt(ax, 1630, 370, "EMA teacher: running average", 17, MUTED, k6, ha="center")
    txt(ax, 1630, 345, "of the student's weights", 17, MUTED, k6, ha="center")
    caption(ax, "CSSL is added on top of this recipe; the recipe alone, on the same splits, is the baseline.",
            seg(t, 1.0, 1.6))


ROWNAMES = [("nucleus", C_NUC), ("rim (≤ 2 px from background)", C_RIM), ("background", C_BG)]
ASSETS = os.environ.get("CSSL_VIDEO_ASSETS", os.path.join(HERE, "assets"))
HAVE_REAL = os.path.exists(os.path.join(ASSETS, "meta.json"))


def _png(name, gray=False):
    from PIL import Image
    im = Image.open(os.path.join(ASSETS, name))
    return np.asarray(im.convert("L" if gray else "RGB")).astype(np.float64) / (1.0 if gray else 255.0)


def _soft(f, k=0.6):
    g = f.mean(-1, keepdims=True)
    return np.clip(0.06 + 0.9 * (g + k * (f - g)), 0, 1)


def _rim3(m, w=2):
    from scipy.ndimage import maximum_filter
    near = maximum_filter(~m, size=2 * w + 1)
    lab = m.astype(np.int64)
    lab[m & near] = 2
    return lab


def _feat_fallback(img, k):
    g = gaussian_filter(img, (1.2 * k, 1.2 * k, 0))[k // 2::k, k // 2::k]
    g = (g - g.min((0, 1))) / (np.ptp(g, (0, 1)) + 1e-9)
    v = g.mean(-1, keepdims=True)
    return np.clip(np.array([0.08, 0.16, 0.32]) + v * np.array([0.30, 0.55, 0.85]) + 0.25 * (g - v), 0, 1)


if HAVE_REAL:
    META = json.load(open(os.path.join(ASSETS, "meta.json")))
    _full, _lab = _png("image.png"), _png("labels.png", gray=True).astype(np.int64)
    cy, cx, cs = META["crop"]
    best, arg = -1, (0, 0)
    for yy in range(0, cs - 255, 32):
        for xx in range(0, cs - 255, 32):
            s = (_lab[cy + yy:cy + yy + 256, cx + xx:cx + xx + 256] > 0).sum()
            if s > best:
                best, arg = s, (yy, xx)
    sy, sx = arg
    IMG = _full[cy + sy:cy + sy + 256, cx + sx:cx + sx + 256]
    LAB = _lab[cy + sy:cy + sy + 256, cx + sx:cx + sx + 256]
    F4 = _soft(_png("feat_l1_clean.png"))[sy // 4:(sy + 256) // 4, sx // 4:(sx + 256) // 4]
    F8 = _soft(_png("feat_l2_clean.png"))[sy // 8:(sy + 256) // 8, sx // 8:(sx + 256) // 8]
    CR_H, CR_HN = _soft(_png("feat_l1_clean.png")), _soft(_png("feat_l1_noisy.png"))
    CR_IMG = _full[cy:cy + cs, cx:cx + cs]
    CR_P0, CR_P1 = _png("pred_clean.png", gray=True) > 127, _png("pred_noisy.png", gray=True) > 127
    REAL_NOTE = "real test image %s, ground-truth labels" % META["image"][:12]
else:
    META = None
    IMG, LAB = TIS, _rim3(TMASK)
    F4, F8 = _feat_fallback(TIS, 4), _feat_fallback(TIS, 8)
    CR_IMG = TIS
    CR_H = _feat_fallback(TIS, 2)
    CR_HN = None
    CR_P0 = CR_P1 = TMASK
    REAL_NOTE = "illustration"
CURVES = json.load(open(os.path.join(ASSETS, "curves.json"))) if os.path.exists(os.path.join(ASSETS, "curves.json")) else None
EMB = json.load(open(os.path.join(ASSETS, "embed.json"))) if os.path.exists(os.path.join(ASSETS, "embed.json")) else None
CLS_MASKS = [LAB == 1, LAB == 2, LAB == 0]
WINS = [(y, x) for y in range(0, 193, 32) for x in range(0, 193, 32)]
PATCHES = []
for _j, (_y, _x) in enumerate(WINS):
    _w = (slice(_y, _y + 64), slice(_x, _x + 64))
    for _r in range(3):
        if CLS_MASKS[_r][_w].sum() >= 25:
            PATCHES.append((_j, _r))
_counts = [0, 0, 0]
SLOT = []
for _j, _r in PATCHES:
    SLOT.append(min(_counts[_r], 19))
    _counts[_r] += 1
N_PER = _counts
FIRST_W = max(range(len(WINS)), key=lambda j: (sum(CLS_MASKS[r][WINS[j][0]:WINS[j][0] + 64, WINS[j][1]:WINS[j][1] + 64].sum() >= 25
                                                   for r in range(3)), -abs(WINS[j][0] - 96) - abs(WINS[j][1] - 96)))


def thumb(j, r, scale=1.0):
    y, x = WINS[j]
    win = (slice(y, y + 64), slice(x, x + 64))
    m = CLS_MASKS[r][win]
    sh = IMG[win] * (0.22 + 0.78 * m[..., None])
    return tint(sh, m, ROWNAMES[r][1], 0.28)


GX, TH, TG = 900, 72, 8
ROW_TOP = [830, 630, 430]


def slot_xy(r, s):
    return GX + (s % 10) * (TH + TG), ROW_TOP[r] - 30 - TH - (s // 10) * (TH + TG)


T_SPLIT, T_SW0, T_SW1, FLY = 3.0, 5.2, 10.4, 0.7
_others = sorted(set(j for j, _ in PATCHES if j != FIRST_W))
PT0 = [T_SPLIT + 1.5 + 0.35 * r if j == FIRST_W else
       T_SW0 + (T_SW1 - T_SW0) * math.sqrt(_others.index(j) / max(1, len(_others) - 1)) for j, r in PATCHES]


def s_patches(ax, t, D):
    header(ax, "Step 1 · class-wise patches", "Every 64 × 64 window gives one patch per class", seg(t, 0, 0.6))
    S, X0, Y0 = 600, 110, 200
    sc = S / 256
    k = seg(t, 0.3, 0.9)
    image(ax, IMG, X0, Y0, S, S, a=k, interp="bilinear")
    scan = seg(t, 0.9, 2.4)
    if scan > 0:
        rows = int(256 * scan)
        ov = np.zeros((256, 256, 4))
        for r, (_, col) in enumerate(ROWNAMES):
            ov[CLS_MASKS[r]] = list(matplotlib.colors.to_rgb(col)) + [0.5 if r < 2 else 0.16]
        ov[rows:] = 0
        ax.imshow(ov, extent=(X0, X0 + S, Y0, Y0 + S), zorder=3.5, interpolation="nearest", aspect="auto",
                  alpha=1 - 0.55 * seg(t, 2.6, 3.4))
        if scan < 1:
            yl = Y0 + S - rows * sc
            ax.plot([X0, X0 + S], [yl, yl], color="white", lw=2.5, alpha=0.9, zorder=6)
            ax.plot([X0, X0 + S], [yl, yl], color=BLUE, lw=12, alpha=0.25, zorder=5.9)
    txt(ax, X0, Y0 - 34, REAL_NOTE, 17, MUTED, k)
    kr = seg(t, 2.4, 3.0)
    for r, (name, col) in enumerate(ROWNAMES):
        yy = ROW_TOP[r] - 20
        txt(ax, GX, yy, name, 20, col, kr, weight="semibold")
        cnt = sum(1 for (j, rr), t0 in zip(PATCHES, PT0) if rr == r and t >= t0 + FLY)
        txt(ax, GX + 10 * (TH + TG) - TG, yy, "%d patches" % cnt, 18, MUTED, kr, ha="right")
    fy, fx = WINS[FIRST_W]
    ks = seg(t, T_SPLIT, T_SPLIT + 0.5)
    if ks > 0 and t < T_SW0 + 0.5:
        a_w = 1 - seg(t, T_SW0, T_SW0 + 0.5)
        bx, by = X0 + fx * sc, Y0 + S - (fy + 64) * sc
        ax.add_patch(Rectangle((bx, by), 64 * sc, 64 * sc, fc=(1, 1, 1, 0.10 * a_w * ks), ec=(1, 1, 1, a_w * ks), lw=3,
                               zorder=7))
        grow = seg(t, T_SPLIT + 0.2, T_SPLIT + 1.0)
        if grow > 0:
            size = 64 * sc + (150 - 64 * sc) * grow
            px, py = bx + (725 - bx) * grow, by + (430 - by) * grow
            rbox(ax, px - 8, py - 8, size + 16, size + 16, fc=SURF, ec=LINE, a=a_w * grow, r=10, z=7.5)
            image(ax, IMG[fy:fy + 64, fx:fx + 64], px, py, size, size, a=a_w, z=8, interp="bilinear", border="white")
            txt(ax, px + size / 2, py - 34, "one window", 17, INK2, a_w * grow, ha="center", z=8)
    landed = {}
    for i, ((j, r), t0) in enumerate(zip(PATCHES, PT0)):
        f = seg(t, t0, t0 + FLY)
        if f <= 0:
            continue
        tx, ty = slot_xy(r, SLOT[i])
        if f >= 1:
            landed[(r, SLOT[i])] = (j, r)
            continue
        y, x = WINS[j]
        if j == FIRST_W:
            sx, sy, ss = 725, 430, 150
        else:
            sx, sy, ss = X0 + x * sc, Y0 + S - (y + 64) * sc, 64 * sc
        lift = math.sin(math.pi * f) * 60
        px, py = sx + (tx - sx) * f, sy + (ty - sy) * f + lift
        size = ss + (TH - ss) * f
        image(ax, thumb(j, r), px, py, size, size, a=1, z=9, interp="bilinear", border=ROWNAMES[r][1])
        if j != FIRST_W and f < 0.25:
            ax.add_patch(Rectangle((sx, sy), ss, ss, fc="none", ec=(1, 1, 1, 1 - 4 * f), lw=2.2, zorder=7))
    for (r, s_), (j, rr) in landed.items():
        tx, ty = slot_xy(r, s_)
        image(ax, thumb(j, rr), tx, ty, TH, TH, a=1, z=6, interp="bilinear", border=ROWNAMES[r][1])
    kf = seg(t, 4.4, 4.9) * (1 - seg(t, 5.4, 5.9))
    txt(ax, 800, 395, "one patch per class", 16, INK2, kf, ha="center", z=8)
    txt(ax, 800, 372, "with ≥ 25 pixels", 16, INK2, kf, ha="center", z=8)
    kd = seg(t, 10.8, 11.4)
    txt(ax, X0 + S / 2, Y0 + S + 18, "%d windows, stride 32 → %d patches" % (len(WINS), len(PATCHES)), 18, INK2, kd,
        ha="center")
    caption(ax, "Patches come from ground truth on labelled images and from confident pseudo-labels on unlabelled ones; "
                "nucleus rims are their own class.", seg(t, 1.0, 1.6))


def plane(ax, img, o, u, v, a=1.0, z=3, border=None, interp="bilinear", lw=1.5):
    if a <= 0.003:
        return
    from matplotlib.transforms import Affine2D
    tr = Affine2D(np.array([[u[0], v[0], o[0]], [u[1], v[1], o[1]], [0, 0, 1]])) + ax.transData
    im = ax.imshow(img, extent=(0, 1, 0, 1), zorder=z, interpolation=interp, alpha=min(1, a))
    im.set_transform(tr)
    if border:
        pts = np.array([o, o + u, o + u + v, o + v])
        ax.add_patch(Polygon(pts, closed=True, fc="none", ec=border, lw=lw, alpha=min(1, a), zorder=z + 0.1))


def P2(o, u, v, fx, fy):
    return np.asarray(o, float) + fx * np.asarray(u, float) + fy * np.asarray(v, float)


def bars(ax, o, cu, cv, hmax, w, a, col, base_rgb):
    from matplotlib.collections import PolyCollection
    if a <= 0.003:
        return
    polys, fcs = [], []
    c = np.array(matplotlib.colors.to_rgb(col))
    n = w.shape[0]
    order = [(i, j) for i in range(n) for j in range(n)]
    order.sort(key=lambda ij: (-(n - 1 - ij[0]), ij[1]))
    for i, j in order:
        p = np.asarray(o, float) + j * np.asarray(cu) + (n - 1 - i) * np.asarray(cv)
        h = np.array([0, hmax * w[i, j]])
        base = np.array(base_rgb[i, j])
        top = base * (1 - w[i, j]) + c * w[i, j]
        q = [p, p + cu, p + cu + cv, p + cv]
        if w[i, j] > 0.01:
            polys += [[q[0], q[1], q[1] + h, q[0] + h], [q[1], q[2], q[2] + h, q[1] + h], [x + h for x in q]]
            fcs += [np.clip(top * 0.62, 0, 1), np.clip(top * 0.45, 0, 1), np.clip(top, 0, 1)]
        else:
            polys.append(q)
            fcs.append(base * 0.9)
    pc = PolyCollection(polys, facecolors=fcs, edgecolors=[(0.07, 0.07, 0.07)], linewidths=0.6, alpha=min(1, a), zorder=6)
    ax.add_collection(pc)


def s_embed(ax, t, D):
    header(ax, "Step 2 · embeddings", "Encode the whole image, then pool its features over the class",
           seg(t, 0, 0.6))
    k = seg(t, 0.3, 0.9)
    fy, fx = WINS[FIRST_W]
    io, iu, iv = np.array([150.0, 250.0]), np.array([330.0, 0.0]), np.array([130.0, 300.0])
    plane(ax, IMG, io, iu, iv, a=k, z=3, border=LINE)
    txt(ax, io[0] + 165, io[1] - 40, "whole image, unmasked", 17, INK2, k, ha="center")
    planes = [(F4, np.array([660.0, 300.0]), np.array([210.0, 0.0]), np.array([82.0, 190.0]), "1/4", 1.2),
              (F8, np.array([1000.0, 340.0]), np.array([150.0, 0.0]), np.array([58.0, 136.0]), "1/8", 2.2)]
    fr = (fx / 256, 1 - (fy + 64) / 256, (fx + 64) / 256, 1 - fy / 256)
    kw = seg(t, 2.8, 3.4)
    corners = [(fr[0], fr[1]), (fr[2], fr[1]), (fr[2], fr[3]), (fr[0], fr[3])]
    if kw > 0:
        pts = np.array([P2(io, iu, iv, a_, b_) for a_, b_ in corners])
        ax.add_patch(Polygon(pts, closed=True, fc=(1, 1, 1, 0.12 * kw), ec=(1, 1, 1, kw), lw=2.5, zorder=7))
    prev = (io, iu, iv)
    for n, (img, o, u, v, lab, t0) in enumerate(planes):
        kp = seg(t, t0, t0 + 0.7)
        for cx_, cy_ in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p, q = P2(*prev, cx_, cy_), P2(o, u, v, cx_, cy_)
            if kp > 0.003:
                ax.plot([p[0], p[0] + (q[0] - p[0]) * kp], [p[1], p[1] + (q[1] - p[1]) * kp], color=LINE, lw=1.2,
                        alpha=0.8, zorder=2)
        plane(ax, img, o, u, v, a=kp, z=3 + n, border=LINE, interp="nearest")
        txt(ax, o[0] + u[0] / 2, o[1] - 34, "encoder features, %s" % lab, 16, MUTED, kp, ha="center")
        if kw > 0 and kp > 0:
            pts = np.array([P2(o, u, v, a_, b_) for a_, b_ in corners])
            ax.add_patch(Polygon(pts, closed=True, fc=(0.22, 0.53, 0.9, 0.3 * kw * kp),
                                 ec=matplotlib.colors.to_rgba(BLUE, kw * kp), lw=2, zorder=7))
            for (a_, b_) in corners:
                p, q = P2(*prev, a_, b_), P2(o, u, v, a_, b_)
                ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=1, alpha=0.6 * kw * kp, zorder=6.5)
        prev = (o, u, v)
    cell = F8[fy // 8:fy // 8 + 8, fx // 8:fx // 8 + 8]
    mfrac = CLS_MASKS[0][fy:fy + 64, fx:fx + 64].astype(float).reshape(8, 8, 8, 8).mean((1, 3))
    kb = seg(t, 4.2, 5.2)
    ob, cu, cv = np.array([1250.0, 250.0]), np.array([34.0, 0.0]), np.array([15.0, 25.0])
    if kb > 0:
        src = P2(planes[1][1], planes[1][2], planes[1][3], fr[0], fr[1])
        for c_ in ((0, 0), (8, 0), (8, 8), (0, 8)):
            q = ob + c_[0] * cu + c_[1] * cv
            ax.plot([src[0], src[0] + (q[0] - src[0]) * kb], [src[1], src[1] + (q[1] - src[1]) * kb], color=BLUE, lw=1,
                    alpha=0.35, zorder=5, ls=(0, (3, 3)))
    grow = seg(t, 5.2, 7.2)
    bars(ax, ob, cu, cv, 150 * grow, mfrac, kb, C_NUC, cell)
    txt(ax, ob[0] + 4 * cu[0] + 4 * cv[0], ob[1] - 40, "8 × 8 feature cells under the window", 16, MUTED, kb, ha="center")
    kt = seg(t, 5.6, 6.2)
    txt(ax, ob[0] + 4 * cu[0] + 4 * cv[0], 640, "bar height = pooling weight m̄", 17, C_NUC, kt, ha="center")
    txt(ax, ob[0] + 4 * cu[0] + 4 * cv[0], 614, "(nucleus fraction of each cell)", 15, MUTED, kt, ha="center")
    kv = seg(t, 7.6, 8.3)
    vx, vy = 1700, 300
    rng = np.random.default_rng(3)
    vec = rng.uniform(0.15, 1, 16)
    if kv > 0:
        for i, v_ in enumerate(vec):
            ax.add_patch(Rectangle((vx, vy + i * 17), 34, 15, fc=(0.23 + 0.5 * v_, 0.31 + 0.3 * v_, 0.85), ec="none",
                                   alpha=kv, zorder=6))
        arrow(ax, (1648, 420), (1692, 420), a=kv, color=BLUE)
        txt(ax, vx + 17, vy - 30, "pooled", 16, INK2, kv, ha="center")
    kz = seg(t, 8.6, 9.3)
    arrow(ax, (1717, 580), (1717, 640), a=kz, color=BLUE)
    if kz > 0:
        ax.add_patch(Circle((1717, 690), 42, fc="none", ec=INK2, lw=1.5, alpha=kz, zorder=6))
        ax.add_patch(Circle((1717 + 26, 690 + 22), 8, fc=C_NUC, ec=BG, lw=1.5, alpha=kz, zorder=7))
        txt(ax, 1717, 748, "z", 28, INK, kz, ha="center", style="italic")
    txt(ax, 1732, 608, "φ, normalise", 15, MUTED, kz, va="center")
    ke = seg(t, 9.3, 10.0)
    txt(ax, 1250, 790, r"$z=\mathrm{normalise}\;\phi\!\left(\frac{\sum_q \bar m_q\, h_q}{\sum_q \bar m_q}\right)$", 28,
        INK, ke, ha="center", va="center")
    kc = seg(t, 0.6, 1.2)
    rbox(ax, 110, 700, 640, 130, fc=SURF, ec=LINE, a=kc, r=14)
    txt(ax, 135, 795, "CHAOS instead: pixel-level masking", 18, INK2, kc, weight="semibold")
    txt(ax, 135, 765, "other pixels of the patch set to 0, then encoded", 15, MUTED, kc)
    mini = IMG[fy:fy + 64, fx:fx + 64] * CLS_MASKS[0][fy:fy + 64, fx:fx + 64, None]
    image(ax, mini, 630, 712, 100, 100, a=kc, border=LINE)
    caption(ax, "The encoder always sees the real image; the class mask only weights the pooling, so context around "
                "small nuclei reaches the embedding.", seg(t, 1.0, 1.6))


RNG6 = np.random.default_rng(11)
NPTS = 26
LAB6 = np.repeat([0, 1, 2], NPTS)
COL6 = [C_NUC, C_RIM, C_BG]


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


CENT3 = _unit(np.array([[0.55, 0.55, 0.62], [-0.15, 0.85, 0.5], [0.1, -0.9, -0.42]]))
Z0 = _unit(RNG6.normal(size=(3 * NPTS, 3)))
Z1 = _unit(CENT3[LAB6] + RNG6.normal(0, 0.2, (3 * NPTS, 3)))
ANC = int(np.argmin(((Z0 - _unit(np.array([0.2, 0.3, 0.93]))) ** 2).sum(1) + 10 * (LAB6 != 0)))
BANK = _unit(CENT3[RNG6.integers(0, 3, 40)] + RNG6.normal(0, 0.22, (40, 3)))
BANK_L = np.argmax(BANK @ CENT3.T, 1)
HN = [int(i) for i in RNG6.choice(np.where(LAB6 != 0)[0], 5, replace=False)]
HN_A = RNG6.uniform(0.15, 0.4, 5)


def slerp(a, b, s):
    s = np.asarray(s, float)[..., None] if np.ndim(s) else s
    dot = np.clip((a * b).sum(-1, keepdims=True), -0.9999, 0.9999)
    om = np.arccos(dot)
    so = np.sin(om)
    return np.sin((1 - s) * om) / so * a + np.sin(s * om) / so * b


def rotm(yaw, pitch):
    cy_, sy_, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry


def proj(p, cx, cy, R, M):
    q = p @ M.T
    f = 3.2 / (3.2 - q[..., 2])
    return np.stack([cx + R * q[..., 0] * f, cy + R * q[..., 1] * f], -1), q[..., 2]


def wire(ax, cx, cy, R, M, a):
    from matplotlib.collections import LineCollection
    if a <= 0.003:
        return
    segs_b, segs_f = [], []
    th = np.linspace(0, 2 * np.pi, 121)
    curves = []
    for lat in np.radians([-60, -30, 0, 30, 60]):
        curves.append(np.stack([np.cos(lat) * np.cos(th), np.full_like(th, np.sin(lat)), np.cos(lat) * np.sin(th)], 1))
    for lon in np.radians(np.arange(0, 180, 30)):
        curves.append(np.stack([np.cos(th) * np.cos(lon), np.sin(th), np.cos(th) * np.sin(lon)], 1))
    for c in curves:
        xy, z = proj(c, cx, cy, R, M)
        for i in range(len(c) - 1):
            (segs_f if z[i] + z[i + 1] > 0 else segs_b).append([xy[i], xy[i + 1]])
    ax.add_collection(LineCollection(segs_b, colors=[LINE], linewidths=0.8, alpha=0.35 * a, zorder=2))
    ax.add_collection(LineCollection(segs_f, colors=[LINE], linewidths=1.1, alpha=0.9 * a, zorder=2.5))
    ax.add_patch(Circle((cx, cy), R * 3.2 / math.sqrt(3.2 ** 2 - 1), fc="none", ec=LINE, lw=1.6, alpha=a, zorder=2.6))


def sil_np(z, lab):
    d = np.abs(z[:, None, :] - z[None, :, :]).sum(-1)
    s = np.zeros(len(z))
    for i in range(len(z)):
        same = lab == lab[i]
        a_ = d[i, same].sum() / max(1, same.sum() - 1)
        b_ = min(d[i, lab == k].mean() for k in np.unique(lab) if k != lab[i])
        s[i] = (b_ - a_) / max(a_, b_, 1e-9)
    return float(s.mean())


def s_contrast(ax, t, D):
    header(ax, "Step 3 · contrastive learning", "Same class together, other classes apart", seg(t, 0, 0.6))
    cx, cy, R = 660, 500, 270
    M = rotm(0.5 + 0.11 * t, -0.35 + 0.05 * math.sin(0.3 * t))
    k = seg(t, 0.3, 1.0)
    wire(ax, cx, cy, R, M, k)
    s = seg(t, 3.0, 9.5)
    Z = _unit(slerp(Z0, Z1, s))
    ka = seg(t, 1.6, 2.2)
    kb = seg(t, 10.6, 11.3)
    allz = Z
    alll = LAB6
    if kb > 0:
        allz = np.concatenate([Z, BANK])
        alll = np.concatenate([LAB6, BANK_L])
    xy, dep = proj(Z, cx, cy, R, M)
    za = Z[ANC]
    if ka > 0:
        fade_l = 1 - seg(t, 9.5, 10.2)
        a_xy = xy[ANC]
        for j in range(len(Z)):
            if j == ANC:
                continue
            same = LAB6[j] == LAB6[ANC]
            ax.plot([a_xy[0], xy[j, 0]], [a_xy[1], xy[j, 1]], color=C_NUC if same else MUTED,
                    lw=1.4 if same else 0.6, alpha=ka * fade_l * (0.6 if same else 0.22) * (0.55 + 0.45 * (dep[j] > 0)),
                    zorder=3)
    if kb > 0:
        bxy, bdep = proj(BANK, cx, cy, R, M)
        order = np.argsort(bdep)
        ax.scatter(bxy[order, 0], bxy[order, 1], s=50 + 30 * (bdep[order] + 1), c=[COL6[l] for l in BANK_L[order]],
                   alpha=0.45 * kb, edgecolors="none", zorder=4)
    order = np.argsort(dep)
    sz = 90 + 90 * (dep[order] + 1)
    cols = [COL6[l] for l in LAB6[order]]
    al = np.clip(0.45 + 0.55 * (dep[order] + 1) / 2, 0, 1) * k
    from matplotlib.colors import to_rgba
    ax.scatter(xy[order, 0], xy[order, 1], s=sz, c=[to_rgba(c_, a_) for c_, a_ in zip(cols, al)],
               edgecolors=[to_rgba(BG, a_) for a_ in al], linewidths=1.5, zorder=5)
    if ka > 0:
        pulse = 1 + 0.15 * math.sin(6 * t)
        ax.add_patch(Circle(xy[ANC], 20 * pulse, fc="none", ec=INK, lw=2.5, alpha=ka, zorder=6))
        txt(ax, xy[ANC, 0] + 26, xy[ANC, 1] + 18, "anchor", 18, INK, ka, z=7)
    kh = seg(t, 7.2, 7.8) * (1 - seg(t, 13.6, 14.2))
    if kh > 0:
        for n_, a_ in zip(HN, HN_A):
            grow = seg(t, 7.4, 9.0)
            arc = _unit(slerp(np.repeat(Z[n_][None], 24, 0), np.repeat(za[None], 24, 0), np.linspace(0, 1, 24)))
            axy, _ = proj(arc, cx, cy, R, M)
            ax.plot(axy[:, 0], axy[:, 1], color=COL6[LAB6[n_]], lw=1.2, ls=(0, (2, 3)), alpha=0.7 * kh, zorder=4)
            mix = _unit(a_ * grow * za + (1 - a_ * grow) * Z[n_])
            mxy, _ = proj(mix[None], cx, cy, R, M)
            ax.add_patch(Circle(mxy[0], 10, fc=BG, ec=COL6[LAB6[n_]], lw=2.6, alpha=kh, zorder=6))
    for i, (name, col) in enumerate((("nucleus", C_NUC), ("rim", C_RIM), ("background", C_BG))):
        if k > 0.003:
            ax.add_patch(Circle((140, 820 - 34 * i), 9, fc=col, ec="none", alpha=k, zorder=5))
        txt(ax, 160, 813 - 34 * i, name, 18, INK2, k)
    txt(ax, 130, 700, "illustration in 3-d", 15, MUTED, k)
    txt(ax, 130, 677, "(real embeddings: 256-d)", 15, MUTED, k)
    if kb > 0:
        rbox(ax, 110, 220, 210, 300, fc=SURF, ec=LINE, a=kb, r=12)
        txt(ax, 128, 485, "memory bank", 18, INK, kb, weight="semibold")
        txt(ax, 128, 460, "4096 past embeddings", 14, MUTED, kb)
        for i in range(12):
            yy = 430 - i * 16 - (t * 40 % 16)
            if yy > 236:
                ax.add_patch(Rectangle((140, yy), 150, 10, fc=COL6[BANK_L[i]], ec="none", alpha=0.55 * kb, zorder=5))
        arrow(ax, (322, 380), (cx - R - 20, cy - 60), a=kb, color=INK2, rad=-0.15)
    ke = seg(t, 1.0, 1.7)
    rbox(ax, 1060, 600, 760, 230, fc=SURF, ec=LINE, a=ke)
    txt(ax, 1090, 790, "Supervised contrastive loss with mixed hard negatives", 19, INK2, ke, weight="semibold")
    txt(ax, 1440, 700, r"$\mathcal{L}_{CL}=-\frac{1}{|P(i)|}\sum_{p\in P(i)}\log\frac{e^{s_{ip}}}"
        r"{e^{s_{ip}}+\sum_{n} e^{s_{in}}+\sum_{t} e^{\tilde s_{it}}}$", 28, INK, ke, ha="center", va="center")
    txt(ax, 1090, 622, "s = cosine similarity / τ,  τ = 0.1", 16, MUTED, ke)
    txt(ax, 1440, 562, r"hollow: $\tilde z=\mathrm{normalise}(\alpha\, z_{anchor}+(1-\alpha)\, z_{neg})$,  $\alpha\sim U(0,\,0.4)$",
        18, INK2, kh, ha="center")
    kg = seg(t, 2.4, 3.0)
    om = sil_np(allz, alll)
    lam = (om + 1) / 2 + 0.05
    rbox(ax, 1060, 190, 760, 320, fc=SURF, ec=BLUE, lw=2, a=kg)
    txt(ax, 1090, 470, "Separation of the points shown", 19, INK2, kg, weight="semibold")
    txt(ax, 1090, 400, "Silhouette  ω = %+.2f" % om, 36, INK, kg, weight="bold")
    bx, bw = 1090, 700
    rbox(ax, bx, 340, bw, 18, fc=SURF2, ec="none", a=kg, r=9)
    zero, end = bx + bw / 2, bx + bw * (om + 1) / 2
    rbox(ax, min(zero, end), 340, abs(end - zero) + 0.1, 18, fc=BLUE, ec="none", a=kg, r=9, z=3)
    for v, lab in ((-1, "−1"), (0, "0"), (1, "1")):
        txt(ax, bx + bw * (v + 1) / 2, 312, lab, 15, MUTED, kg, ha="center")
    txt(ax, 1090, 262, "noise level for the consistency loss", 18, INK2, kg)
    txt(ax, 1090, 220, "λ = (ω + 1)/2 + 0.05 = %.2f" % lam, 26, BLUE, kg, weight="bold")
    caps = [(0, "Patch embeddings of the batch: nucleus, rim and background, all mixed at first."),
            (3.0, "Positives are pulled to the anchor, negatives pushed away; every patch is an anchor."),
            (7.2, "Negatives are moved towards the anchor by a random α, giving harder negatives."),
            (10.6, "On MoNuSeg, 4096 embeddings from earlier batches join as extra positives and negatives.")]
    cur = max(i for i, (t0, _) in enumerate(caps) if t >= t0)
    t0 = caps[cur][0]
    caption(ax, caps[cur][1], seg(t, t0 + 0.2, t0 + 0.7) if cur else seg(t, 1.0, 1.6))


def _lam_curve():
    if CURVES and "monuseg_s0" in CURVES:
        c = CURVES["monuseg_s0"]
        it = np.array([x if x is not None else np.nan for x in c["iter"]], float)
        lam = np.array([x if x is not None else np.nan for x in c["lam"]], float)
        om = np.array([x if x is not None else np.nan for x in c["omega"]], float)
        ok = np.isfinite(it) & np.isfinite(lam)
        return it[ok], lam[ok], om[ok], True
    it = np.linspace(2000, 30000, 300)
    lam = 0.55 + 0.34 * (1 - np.exp(-(it - 2000) / 6000))
    return it, lam, 2 * (lam - 0.05) - 1, False


LC_IT, LC_LAM, LC_OM, LC_REAL = _lam_curve()


def s_consistency(ax, t, D):
    header(ax, "Step 4 · Silhouette-adaptive consistency", "Perturb the features only as hard as the embedding allows",
           seg(t, 0, 0.6))
    k = seg(t, 0.3, 0.9)
    s = seg(t, 1.2, 8.2)
    n = max(1, int(round(s * (len(LC_IT) - 1))))
    lam_now = float(LC_LAM[n])
    lam_end = float(LC_LAM[-1])
    rbox(ax, 110, 230, 470, 590, fc=SURF, ec=LINE, a=k)
    txt(ax, 135, 780, "λ during training" + (" (MoNuSeg, seed 0)" if LC_REAL else ""), 17, INK2, k,
        weight="semibold")
    gx, gy, gw, gh = 150, 420, 400, 300
    lo, hi = 0.5, 1.05
    if k > 0.003:
        ax.plot([gx, gx + gw], [gy, gy], color=LINE, lw=1.2, alpha=k, zorder=3)
        ax.plot([gx, gx], [gy, gy + gh], color=LINE, lw=1.2, alpha=k, zorder=3)
        X = gx + gw * LC_IT / max(LC_IT.max(), 1)
        Y = gy + gh * (LC_LAM - lo) / (hi - lo)
        ax.plot(X[:n + 1], Y[:n + 1], color=BLUE, lw=3, alpha=k, zorder=4)
        ax.add_patch(Circle((X[n], Y[n]), 8, fc=BLUE, ec=BG, lw=2, alpha=k, zorder=5))
        for v in (0.6, 0.8, 1.0):
            txt(ax, gx - 10, gy + gh * (v - lo) / (hi - lo), "%.1f" % v, 13, MUTED, k, ha="right", va="center")
        txt(ax, gx + gw, gy - 26, "iteration %dk" % int(round(LC_IT.max() / 1000)), 13, MUTED, k, ha="right")
        txt(ax, gx, gy - 26, "0", 13, MUTED, k)
    txt(ax, 135, 350, "λ = %.2f" % lam_now, 34, BLUE, k, weight="bold")
    txt(ax, 135, 300, r"$\lambda=\frac{\bar\omega+1}{2}+0.05$,  $\bar\omega$: running mean of ω", 17, INK2, k)
    txt(ax, 135, 262, "mixed → mild noise; separated → stronger", 15, MUTED, k)
    S = 300
    img_h = CR_H
    image(ax, img_h, 650, 440, S, S, a=k, interp="nearest", border=LINE)
    txt(ax, 650 + S / 2, 405, "encoder features h (1/4)", 18, INK2, k, ha="center")
    arrow(ax, (962, 590), (1022, 590), a=k)
    if CR_HN is not None:
        noisy = np.clip(CR_H + math.sqrt(lam_now / max(lam_end, 1e-6)) * (CR_HN - CR_H), 0, 1)
    else:
        rng = np.random.default_rng(1)
        noisy = np.clip(CR_H + math.sqrt(lam_now) * 0.35 * rng.normal(0, 1, CR_H.shape), 0, 1)
    image(ax, noisy, 1035, 440, S, S, a=k, interp="nearest", border=BLUE)
    txt(ax, 1035 + S / 2, 405, r"$\tilde h = h+\sqrt{\lambda}\,\sigma(h)\,\xi$", 24, INK, k, ha="center")
    arrow(ax, (1347, 590), (1395, 590), a=k)
    trapezoid(ax, 1400, 525, 100, 130, k, "dec.")
    arrow(ax, (1505, 590), (1545, 590), a=k)
    kp = seg(t, 8.4, 9.0)
    pv = mask_img(CR_P1, C_NUC)
    if kp > 0:
        diff = CR_P1 != CR_P0
        pv[diff] = np.array(matplotlib.colors.to_rgb(YELLOW))
    image(ax, pv if kp > 0 else mask_img(CR_P0, C_NUC) * 0.4, 1560, 490, 200, 200, a=k, border=LINE, interp="nearest")
    if META:
        txt(ax, 1660, 455, "prediction from perturbed features", 15, INK2, kp, ha="center")
        txt(ax, 1660, 432, "agrees with the clean one: %.1f%% F1" % META["agreement"], 14, BLUE, kp, ha="center",
            weight="semibold")
        txt(ax, 1660, 409, "yellow: changed pixels (λ = %.2f)" % META["lambda_final"], 14, MUTED, kp, ha="center")
    else:
        txt(ax, 1660, 455, "must still match the", 16, INK2, k, ha="center")
        txt(ax, 1660, 432, "teacher's pseudo-label", 16, INK2, k, ha="center")
    kk = seg(t, 4.5, 5.1)
    rbox(ax, 650, 250, 1160, 90, fc=SURF, ec=LINE, a=kk, r=14)
    txt(ax, 1230, 295, r"$\mathcal{L}_{CR}=\frac{1}{2}\left[\mathrm{CE}_M(g(\tilde h),\hat y)+\mathrm{Dice}_M(g(\tilde h),\hat y)\right]$"
        "     every encoder level, confident pixels only", 21, INK, kk, ha="center", va="center")
    note = ("real features of the CSSL network on a test image; noise ξ fixed, scaled by √λ" if META
            else "illustration")
    txt(ax, 650, 770, note, 15, MUTED, k)
    caption(ax, "Robustness to feature noise is demanded only once the embedding separates the classes.",
            seg(t, 1.0, 1.6))


EQ_PIECES = [(r"$\mathcal{L}\;=\;\frac{1}{2}\,($", INK2, 0.6),
             (r"$\mathcal{L}_{sup}+\frac{1}{2}\mathcal{L}_{s_1}+\frac{1}{2}\mathcal{L}_{s_2}$", INK2, 0.6),
             (r"$\;+\;\beta\,\mathcal{L}_{CR}$", BLUE, 1.6), (r"$)$", INK2, 0.6),
             (r"$\;+\;w(t)\,\mathcal{L}_{CL}$", BLUE, 2.2)]
_EQ = {}
LOOP = [("labelled +\nunlabelled batch", False), ("teacher:\npseudo-labels", False), ("recipe losses\nL_sup, L_s1, L_s2", False),
        ("class-wise\npatches", True), ("embeddings z", True), ("contrastive\nloss L_CL", True), ("Silhouette ω", True),
        ("noise level λ", True), ("consistency\nloss L_CR", True), ("SGD step,\nEMA teacher", False)]


def s_objective(ax, t, D):
    header(ax, "Putting it together", "One training step, one objective", seg(t, 0, 0.6))
    size = 50
    if "w" not in _EQ:
        r = ax.figure.canvas.get_renderer()
        ws = []
        for p, _, _ in EQ_PIECES:
            tt = ax.text(0, 0, p, fontsize=size)
            ws.append(tt.get_window_extent(renderer=r).width)
            tt.remove()
        _EQ["w"] = ws
    x = (W - sum(_EQ["w"])) / 2
    for (p, col, t0), w_ in zip(EQ_PIECES, _EQ["w"]):
        txt(ax, x, 780, p, size, col, seg(t, t0, t0 + 0.6))
        x += w_
    txt(ax, W / 2, 712, "gray: the weak-to-strong recipe      blue: CSSL      β = 0.5, w = 0.1; β = w = 0 gives the recipe",
        19, MUTED, seg(t, 2.6, 3.2), ha="center")
    cx, cy, rx, ry = W / 2, 400, 700, 190
    n = len(LOOP)
    kl = seg(t, 3.0, 3.8)
    th = np.linspace(0, 2 * np.pi, 200)
    if kl > 0:
        ax.plot(cx + rx * np.cos(th), cy + ry * np.sin(th), color=LINE, lw=2, alpha=kl, zorder=2)
    pos = [(cx + rx * math.cos(math.pi / 2 - 2 * math.pi * i / n), cy + ry * math.sin(math.pi / 2 - 2 * math.pi * i / n))
           for i in range(n)]
    ph = ((t - 4.0) / 0.55) if t > 4.0 else -1
    for i, ((name, css), (px, py)) in enumerate(zip(LOOP, pos)):
        ki = seg(t, 3.2 + 0.08 * i, 3.7 + 0.08 * i)
        lit = 0.0
        if ph >= 0:
            d = (ph - i) % n
            lit = max(0.0, 1 - d / 1.5) if d < 1.5 else 0.0
        col = BLUE if css else INK2
        fc = SURF2 if not lit else ("#1b2a3d" if css else "#2e2e2b")
        rbox(ax, px - 110, py - 42, 220, 84, fc=fc, ec=col if (css or lit) else LINE, lw=1.6 + 1.6 * lit, a=ki, r=14,
             z=4)
        txt(ax, px, py, name, 16, INK if lit or css else INK2, ki, ha="center", va="center", z=5,
            weight="semibold" if css else "regular", linespacing=1.15)
    if ph >= 0:
        a_ = 2 * math.pi * ((ph % n) / n)
        bx_, by_ = cx + rx * math.cos(math.pi / 2 - a_), cy + ry * math.sin(math.pi / 2 - a_)
        ax.add_patch(Circle((bx_, by_), 14, fc=YELLOW, ec="none", alpha=0.9, zorder=3))
        ax.add_patch(Circle((bx_, by_), 28, fc=YELLOW, ec="none", alpha=0.18, zorder=3))
        txt(ax, cx, cy + 14, "one iteration", 24, INK2, seg(t, 4.0, 4.6), ha="center", va="center", weight="semibold")
        txt(ax, cx, cy - 22, "repeated 30,000 times on MoNuSeg", 17, MUTED, seg(t, 4.0, 4.6), ha="center", va="center")
    caption(ax, "", 0)


def trapezoid(ax, x, y, w, h, a, label, color=SURF2):
    if a <= 0.003:
        return
    ax.add_patch(Polygon([(x, y), (x + w, y + h * 0.25), (x + w, y + h * 0.75), (x, y + h)], closed=True, fc=color,
                         ec=LINE, lw=1.6, alpha=a, zorder=3))
    txt(ax, x + w / 2, y + h / 2, label, 20, INK, a, ha="center", va="center", weight="semibold")


def s_learned(ax, t, D):
    header(ax, "Real embeddings · MoNuSeg test images, seed 0", "What the contrastive branch does to the encoder",
           seg(t, 0, 0.6))
    Em = EMB["models"]
    lab = np.array(EMB["label"])
    keys = [("random weights", "init"), ("recipe only", "recipe"), ("CSSL", "cssl")]
    XY = [np.array(Em[k]["xy"]) for _, k in keys]
    s1, s2 = seg(t, 2.6, 4.6), seg(t, 6.2, 8.4)
    P = XY[0] + (XY[1] - XY[0]) * s1 if s2 <= 0 else XY[1] + (XY[2] - XY[1]) * s2
    stage = 0 if s1 < 0.5 else (1 if s2 < 0.5 else 2)
    ox, oy, sw, sh = 110, 200, 880, 520
    cmap = {"nucleus": C_NUC, "rim": C_RIM, "background": C_BG}
    ccol = [cmap[c] for c in EMB["classes"]]
    k = seg(t, 0.3, 1.0)
    rbox(ax, ox, oy, sw, sh, fc=SURF, ec=LINE, a=k, r=16)
    lim = np.percentile(np.abs(np.concatenate(XY)), 99.5)
    X = np.clip(ox + sw / 2 + P[:, 0] / lim * (sw / 2 - 30), ox + 10, ox + sw - 10)
    Y = np.clip(oy + sh / 2 + P[:, 1] / lim * (sh / 2 - 30), oy + 10, oy + sh - 10)
    perm = np.random.default_rng(0).permutation(len(lab))
    from matplotlib.colors import to_rgba
    cols = [to_rgba(ccol[c], 0.8 * k) for c in lab[perm]]
    if k > 0.003:
        ax.scatter(X[perm], Y[perm], s=14, c=cols, edgecolors="none", zorder=4)
    for i, (name, key) in enumerate(keys):
        on = i == stage
        bx = ox + i * (sw / 3)
        rbox(ax, bx + 6, oy + sh + 18, sw / 3 - 12, 84, fc="#1b2a3d" if on else SURF2, ec=BLUE if on else LINE,
             lw=2.2 if on else 1.2, a=k, r=12)
        txt(ax, bx + sw / 6, oy + sh + 74, name, 19, BLUE if on else INK2, k, ha="center", weight="semibold")
        txt(ax, bx + sw / 6, oy + sh + 34, "ω = %.2f" % Em[key]["silhouette_l1"], 26 if on else 22, INK if on else MUTED,
            k, ha="center", weight="bold" if on else "regular")
    lx = ox + 30
    for i, name in enumerate(EMB["classes"]):
        if k > 0.003:
            ax.add_patch(Circle((lx, oy - 30), 8, fc=ccol[i], ec="none", alpha=k, zorder=5))
        txt(ax, lx + 16, oy - 37, name, 17, INK2, k)
        lx += 70 + 11 * len(name)
    txt(ax, ox + sw, oy - 37, "%d test patches, ground-truth classes" % len(lab), 15, MUTED, k, ha="right")
    if CURVES and "monuseg_s0" in CURVES:
        c = CURVES["monuseg_s0"]
        it = np.array([np.nan if v is None else v for v in c["iter"]], float)
        om = np.array([np.nan if v is None else v for v in c["omega"]], float)
        la = np.array([np.nan if v is None else v for v in c["lam"]], float)
        gx, gy, gw, gh = 1130, 250, 650, 430
        kc = seg(t, 1.2, 1.8)
        rbox(ax, 1060, 200, 750, 622, fc=SURF, ec=LINE, a=kc, r=16)
        txt(ax, 1090, 785, "During training (CSSL, seed 0)", 19, INK2, kc, weight="semibold")
        lo, hi = -0.2, 1.0
        Xc = gx + gw * it / np.nanmax(it)
        Yo, Yl = gy + gh * (om - lo) / (hi - lo), gy + gh * (la - lo) / (hi - lo)
        if kc > 0.003:
            for v in (0.0, 0.5, 1.0):
                yy = gy + gh * (v - lo) / (hi - lo)
                ax.plot([gx, gx + gw], [yy, yy], color=LINE, lw=0.8, alpha=kc, zorder=2)
                txt(ax, gx - 10, yy, "%.1f" % v, 13, MUTED, kc, ha="right", va="center")
            txt(ax, gx + gw, gy - 30, "iteration %dk" % int(round(np.nanmax(it) / 1000)), 13, MUTED, kc, ha="right")
        prog = seg(t, 1.8, 10.5)
        n = int(prog * (len(it) - 1))
        if n > 1:
            ax.plot(Xc[:n], Yo[:n], color=VIOLET, lw=1.0, alpha=0.55 * kc, zorder=3)
            ax.plot(Xc[:n], Yl[:n], color=BLUE, lw=3.0, alpha=kc, zorder=4)
            if np.isfinite(Yl[n - 1]):
                ax.add_patch(Circle((Xc[n - 1], Yl[n - 1]), 8, fc=BLUE, ec=BG, lw=2, alpha=kc, zorder=5))
        txt(ax, 1090, 748, "ω, Silhouette of each batch's patch embeddings", 16, VIOLET, kc)
        txt(ax, 1090, 722, "λ, noise level of the consistency loss", 16, BLUE, kc, weight="semibold")
    caption(ax, "t-SNE of real 1/8-resolution encoder features, pooled per patch; ω is computed in the full feature "
                "space, not in the 2-d map.", seg(t, 1.0, 1.6))


def s_fair(ax, t, D):
    header(ax, "Evaluation", "Protocol", seg(t, 0, 0.6))
    items = [
        ("Baseline protocols", "backbone, splits, labelled fraction, metric and tested checkpoint (Self-Loop; DAFNet)"),
        ("Supervised reference", "our supervised model %.2f vs. Self-Loop's supervised %.2f on MoNuSeg" % (NUM["m_sup"], NUM["sl_sup"])),
        ("Recipe-only baseline", "same code, data and splits, CSSL components switched off"),
        ("Testing", "single models, no test-time augmentation or post-processing"),
        ("Comparison rule", "higher mean and higher on at least 2 of 3 splits"),
        ("Train / test separation", "no image or subject in both training and test (file names and pixel hashes)"),
    ]
    for i, (a_, b_) in enumerate(items):
        k = seg(t, 0.8 + 1.15 * i, 1.4 + 1.15 * i)
        y = 800 - i * 105
        tick(ax, 150, y + 10, 22, BLUE, k)
        txt(ax, 200, y + 14, a_, 27, INK, k, weight="bold")
        txt(ax, 200, y - 22, b_, 20, INK2, k)
    caption(ax, "", 0)


def axes_frame(ax, x0, y0, w, h, lo, hi, ticks, a, ylabel):
    if a <= 0.003:
        return
    for v in ticks:
        y = y0 + h * (v - lo) / (hi - lo)
        ax.plot([x0, x0 + w], [y, y], color=LINE, lw=1, alpha=a, zorder=1)
        txt(ax, x0 - 16, y, "%g" % v, 17, MUTED, a, ha="right", va="center")
    txt(ax, x0 - 75, y0 + h / 2, ylabel, 18, MUTED, a, ha="center", va="center", rotation=90)


def s_res_monuseg(ax, t, D):
    header(ax, "Results · MoNuSeg, 50% labels", "Self-Loop's protocol: ResUNet-18, 15 of 30 images labelled",
           seg(t, 0, 0.6))
    x0, y0, w, h, lo, hi = 260, 230, 760, 600, 75, 82
    k = seg(t, 0.3, 0.9)
    axes_frame(ax, x0, y0, w, h, lo, hi, range(75, 83), k, "nuclei F1 (%)")

    def Y(v):
        return y0 + h * (v - lo) / (hi - lo)

    kl = seg(t, 1.0, 2.0)
    if kl > 0:
        ax.plot([x0, x0 + w * kl], [Y(NUM["sl"])] * 2, color=INK, lw=2, ls=(0, (6, 4)), zorder=3)
        txt(ax, x0 + 10, Y(NUM["sl"]) + 12, "Self-Loop (published) %.2f" % NUM["sl"], 18, INK, kl)
    groups = [("supervised", [NUM["m_sup"]], SUPC, 2.2), ("recipe only", NUM["m_rec"], ORANGE, 3.2),
              ("CSSL", NUM["m_cssl"], BLUE, 4.6)]
    for g, (name, vals, col, t0) in enumerate(groups):
        gx = x0 + w * (g + 0.5) / 3
        txt(ax, gx, y0 - 40, name, 20, INK if g == 2 else INK2, seg(t, t0 - 0.4, t0), ha="center",
            weight="bold" if g == 2 else "regular")
        for i, v in enumerate(vals):
            kk = seg(t, t0 + 0.2 * i, t0 + 0.2 * i + 0.5)
            if kk > 0.003:
                off = (i - (len(vals) - 1) / 2) * 34
                ax.add_patch(Circle((gx + off, Y(v) + 60 * (1 - kk)), 13, fc=col, ec=BG, lw=2, alpha=kk, zorder=5))
        km = seg(t, t0 + 0.8, t0 + 1.2)
        mu = mean(vals)
        if len(vals) > 1 and km > 0.003:
            ax.plot([gx - 70, gx + 70], [Y(mu)] * 2, color=col, lw=5, alpha=km, solid_capstyle="round", zorder=4)
        lab = "%.2f" % mu + (" ± %.2f" % np.std(vals) if len(vals) > 1 else "")
        txt(ax, gx + 82, Y(mu), lab, 18, INK, km, va="center", weight="bold" if g == 2 else "regular")
    kb = seg(t, 6.4, 7.0)
    rbox(ax, 1180, 250, 640, 580, fc=SURF, ec=BLUE, lw=2, a=kb)
    txt(ax, 1220, 690, "%.2f" % mean(NUM["m_cssl"]), 110, BLUE, kb, weight="bold")
    txt(ax, 1225, 630, "± %.2f nuclei F1, 3 label subsets" % np.std(NUM["m_cssl"]), 22, INK2, kb)
    lines = [("%s over the identical recipe" % sgn(m_gain), "higher on %d of 3 subsets" % m_wins, 7.2),
             ("%s over published Self-Loop" % sgn(mean(NUM["m_cssl"]) - NUM["sl"]), "its F1 read as nuclei F1; all 3 subsets", 7.9),
             ("above supervised with 100% labels", "%.2f (one seed); CSSL uses half the labels" % NUM["m_sup100"], 8.6)]
    for i, (a_, b_, t0) in enumerate(lines):
        kk = seg(t, t0, t0 + 0.5)
        tick(ax, 1240, 548 - i * 100, 16, BLUE, kk)
        txt(ax, 1272, 545 - i * 100, a_, 21, INK, kk, weight="semibold")
        txt(ax, 1272, 515 - i * 100, b_, 18, MUTED, kk)
    caption(ax, "The recipe alone does not beat Self-Loop on average; CSSL does, on every label subset.",
            seg(t, 1.0, 1.6))


def s_res_chaos(ax, t, D):
    header(ax, "Results · CHAOS T2, 25% labels", "DAFNet's protocol: its splits, U-Net, labelled subjects and Dice formula",
           seg(t, 0, 0.6))
    x0, y0, w, h, lo, hi = 260, 230, 760, 600, 76, 94
    k = seg(t, 0.3, 0.9)
    axes_frame(ax, x0, y0, w, h, lo, hi, range(76, 95, 2), k, "Dice (%), best-validation checkpoint")

    def Y(v):
        return y0 + h * (v - lo) / (hi - lo)

    kl = seg(t, 1.0, 2.0)
    if kl > 0:
        ax.plot([x0, x0 + w * kl], [Y(NUM["daf"])] * 2, color=INK, lw=2, ls=(0, (6, 4)), zorder=3)
        txt(ax, x0 + 10, Y(NUM["daf"]) - 30, "DAFNet (published) %.1f" % NUM["daf"], 18, INK, kl)
    rows = [("supervised", NUM["c_sup"], SUPC, 2.0), ("recipe only", NUM["c_rec"], ORANGE, 2.8),
            ("CSSL", NUM["c_cssl"], BLUE, 3.6)]
    for c in range(4):
        txt(ax, x0 + w * (c + 0.5) / 4, y0 - 40, ["split 0", "split 1", "split 2", "mean"][c], 20, INK2, k, ha="center")
    for j, (name, vals, col, t0) in enumerate(rows):
        for c in range(4):
            v = vals[c] if c < 3 else mean(vals)
            gx = x0 + w * (c + 0.5) / 4 + (j - 1) * 38
            kk = seg(t, t0 + 0.12 * c, t0 + 0.12 * c + 0.5)
            if kk > 0.003:
                ax.add_patch(Circle((gx, Y(v) + 50 * (1 - kk)), 13 if c < 3 else 17, fc=col, ec=BG, lw=2, alpha=kk,
                                    zorder=5))
        if k > 0.003:
            ax.add_patch(Circle((x0 + 30 + 200 * j, y0 + h + 30), 9, fc=col, ec="none", alpha=k, zorder=5))
        txt(ax, x0 + 48 + 200 * j, y0 + h + 23, name, 18, INK2, k)
    kb = seg(t, 4.6, 5.2)
    rbox(ax, 1180, 560, 640, 270, fc=SURF, ec=BLUE, lw=2, a=kb)
    txt(ax, 1220, 700, "%.2f" % mean(NUM["c_cssl"]), 96, BLUE, kb, weight="bold")
    txt(ax, 1225, 640, "Dice, %s over DAFNet (%.1f)" % (sgn(mean(NUM["c_cssl"]) - NUM["daf"]), NUM["daf"]), 22, INK2, kb)
    txt(ax, 1225, 600, "above it on all 3 splits", 18, MUTED, kb)
    ko = seg(t, 6.3, 7.0)
    rbox(ax, 1180, 230, 640, 300, fc="#2a1f17", ec=ORANGE, lw=2, a=ko)
    txt(ax, 1210, 470, "Comparison with the recipe", 22, ORANGE, ko, weight="bold")
    txt(ax, 1210, 410, "Recipe alone: %.2f Dice" % mean(NUM["c_rec"]), 22, INK, ko)
    txt(ax, 1210, 360, "CSSL − recipe: %s Dice" % sgn(c_gain_b), 19, INK, ko)
    txt(ax, 1210, 300, "On par with the recipe on CHAOS.", 19, INK2, ko)


def s_ablation(ax, t, D):
    header(ax, "Ablations", "Remove one component at a time (one split each)", seg(t, 0, 0.6))
    A, B = M["ablation_s0"], C["ablation_s1"]
    left = [("contrastive loss", A["Full"] - A["w/o CL"]), ("boundary class", A["Full"] - A["w/o boundary class"]),
            ("feature-level masking", A["Full"] - A["w/o feature-level embedding"]), ("hard negatives", A["Full"] - A["w/o HNM"]),
            ("consistency loss", A["Full"] - A["w/o CR"]), ("Silhouette feedback", A["Full"] - A["w/o Silhouette"]),
            ("memory bank", A["Full"] - A["w/o memory bank"])]
    right = [("contrastive loss", B["Full"] - B["w/o CL"]), ("Silhouette feedback", B["Full"] - B["w/o Silhouette"]),
             ("hard negatives", B["Full"] - B["w/o HNM"]), ("consistency loss", B["Full"] - B["w/o CR"])]
    for p, (title, rows, x0, scale) in enumerate((("MoNuSeg, seed 0 (full %.2f)" % A["Full"], left, 470, 110),
                                                  ("CHAOS, split 1 (full %.2f)" % B["Full"], right, 1340, 48))):
        k = seg(t, 0.4 + 0.6 * p, 1.0 + 0.6 * p)
        txt(ax, x0 - 330, 820, title, 22, INK, k, weight="bold")
        txt(ax, x0 - 330, 790, "points lost when the component is removed", 16, MUTED, k)
        if k > 0.003:
            ax.plot([x0, x0], [215, 760], color=MUTED, lw=1.2, alpha=k, zorder=3)
        for i, (name, d) in enumerate(rows):
            y = 720 - i * 75
            kk = seg(t, 1.2 + 0.25 * i + 0.6 * p, 2.2 + 0.25 * i + 0.6 * p)
            txt(ax, x0 - 18 + min(0.0, d * scale), y, name, 19, INK2, k, ha="right", va="center")
            wpx = d * scale * kk
            rbox(ax, min(x0, x0 + wpx), y - 20, abs(wpx) + 0.1, 40, fc=BLUE if d > 0 else RED, ec="none", a=k, r=6, z=4)
            txt(ax, x0 + max(wpx, 0) + 12, y, sgn(d) if kk > 0.95 else "", 18, INK, k, va="center")
    kt = seg(t, 6.0, 6.6)
    txt(ax, W / 2, 190, "The contrastive branch is the engine.", 22, INK, kt, ha="center", weight="semibold")
    txt(ax, W / 2, 156, "Consistency helps little (and hurts on CHAOS split 1); the memory bank lowered the score on MoNuSeg.", 18, INK2, kt, ha="center")
    caption(ax, "Blue: the component helps. Red: the score is higher without it.",
            seg(t, 1.0, 1.6))


def s_outro(ax, t, D):
    header(ax, "Summary", "", seg(t, 0, 0.6))
    pts = ["MoNuSeg 50%%: CSSL %.2f nuclei F1" % mean(NUM["m_cssl"]),
           "Self-Loop %.2f, recipe %.2f" % (NUM["sl"], mean(NUM["m_rec"])),
           "CHAOS T2 25%%: CSSL %.2f Dice" % mean(NUM["c_cssl"]),
           "DAFNet %.1f, recipe %.2f" % (NUM["daf"], mean(NUM["c_rec"]))]
    for i, p in enumerate(pts):
        k = seg(t, 0.4 + 0.4 * i, 1.0 + 0.4 * i)
        txt(ax, 110, 840 - i * 56, p, 25, INK if i % 2 == 0 else INK2, k, weight="semibold" if i % 2 == 0 else "regular")
    k = seg(t, 2.4, 3.0)
    rbox(ax, 110, 220, 1700, 320, fc="#0b0b0a", ec=LINE, a=k, r=18)
    txt(ax, 140, 495, "Code", 20, MUTED, k)
    code = ["$ bash setup/install_env.sh && conda activate cssl",
            "$ DATA=/data/med STAGES=setup bash run_paper.sh",
            "$ DATA=/data/med bash run_paper.sh"]
    for i, c in enumerate(code):
        txt(ax, 140, 430 - i * 62, c, 26, AQUA if i < 2 else INK, seg(t, 3.0 + 0.4 * i, 3.5 + 0.4 * i),
            family="DejaVu Sans Mono")
    caption(ax, "", 0)


FIG = os.path.join(os.path.dirname(HERE), "figures")


def _panels(name, row, cols):
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(FIG, name)).convert("RGB")).astype(np.float64) / 255.0
    r0, r1 = row
    return [im[r0:r1, c0:c1] for c0, c1 in cols]


QM = json.load(open(os.path.join(FIG, "fig_qual_monuseg.json")))
QM_I = QM["test_names"].index(QM["images"][0])
QM_S = [QM["per_image_nuclei_f1"][k][QM_I] for k in ("Supervised", "Recipe only", "CSSL (ours)")]
PM = _panels("fig_qual_monuseg.png", (59, 295), [(77, 314), (321, 558), (564, 801), (808, 1045), (1052, 1289)])
PC = _panels("fig_qual_chaos.png", (363, 600), [(57, 294), (301, 538), (544, 781), (788, 1025), (1031, 1268)])
QC_S = [18.0, 27.1, 43.4]
TPc, FPc, FNc = (0.93, 0.63, 0.0), (0.89, 0.29, 0.28), (0.38, 0.31, 0.84)
ORG = [("liver", (0.93, 0.63, 0.0)), ("right kidney", (0.91, 0.48, 0.64)), ("left kidney", (0.38, 0.31, 0.84)),
       ("spleen", (0.0, 0.51, 0.0))]


def _wipe(ax, imgs, t, t0, gap, x, y, S, a):
    k = 0
    for i in range(1, len(imgs)):
        if t >= t0 + gap * i:
            k = i
    image(ax, imgs[k], x, y, S, S, a=a, interp="bilinear", border=LINE)
    if k + 1 < len(imgs):
        f = seg(t, t0 + gap * (k + 1) - 0.6, t0 + gap * (k + 1))
        if f > 0.001:
            n = imgs[k + 1]
            cw = max(1, int(round(n.shape[1] * f)))
            image(ax, n[:, :cw], x, y, S * cw / n.shape[1], S, a=a, interp="bilinear")
            ax.plot([x + S * cw / n.shape[1]] * 2, [y, y + S], color="white", lw=3, alpha=a, zorder=5)
    return k


def s_real(ax, t, D):
    half = 8.6
    part = 0 if t < half else 1
    tt = t if part == 0 else t - half
    a = min(1.0, tt / 0.4, (half - tt) / 0.4) if part == 0 else min(1.0, tt / 0.4)
    a = max(0.0, a)
    if part == 0:
        header(ax, "Real test images · MoNuSeg, seed 0",
               "Test image with the largest per-image gain (1 of 14)", 1.0)
        P, scores, unit = PM, QM_S, "nuclei F1 (whole image)"
        names = ["Image", "Prediction", "Ground truth"]
        legend = [("nucleus, correct", TPc), ("false positive", FPc), ("missed nucleus", FNc)]
        cap = "The supervised model and the recipe mark large background regions as nuclei; CSSL does not."
    else:
        header(ax, "Real test images · CHAOS T2, split 1",
               "Test slice with the largest organ area", 1.0)
        P, scores, unit = PC, QC_S, "Dice on this slice"
        names = ["T2 image", "Prediction", "Ground truth"]
        legend = ORG
        cap = "CSSL recovers more of the liver and part of the spleen than the recipe."
    S, y = 440, 360
    xs = [170, 740, 1310]
    image(ax, P[0], xs[0], y, S, S, a=a, interp="bilinear", border=LINE)
    image(ax, P[1], xs[2], y, S, S, a=a, interp="bilinear", border=LINE)
    k = _wipe(ax, P[2:], tt, 0.9, 2.3, xs[1], y, S, a)
    for x, n in zip(xs, names):
        txt(ax, x + S / 2, y + S + 22, n, 22, INK2, a, ha="center")
    models = [("Supervised", SUPC), ("Recipe only", ORANGE), ("CSSL", BLUE)]
    for i, (m, col) in enumerate(models):
        ki = seg(tt, 0.9 + 2.3 * i - 0.3, 0.9 + 2.3 * i + 0.2)
        if ki <= 0.003:
            continue
        on = i == k
        bx = xs[1] + i * (S / 3)
        rbox(ax, bx + 6, 250, S / 3 - 12, 78, fc=SURF if not on else "#1b2a3d" if col == BLUE else SURF2,
             ec=col if on else LINE, lw=2.2 if on else 1.2, a=a * ki, r=12)
        txt(ax, bx + S / 6, 302, m, 17, col if on else MUTED, a * ki, ha="center", weight="semibold")
        txt(ax, bx + S / 6, 266, "%.1f" % scores[i], 26 if on else 22, INK if on else MUTED, a * ki, ha="center",
            weight="bold" if on else "regular")
    txt(ax, xs[1] + S / 2, 222, unit, 16, MUTED, a, ha="center")
    lx = 140
    for name, col in legend:
        if a > 0.003:
            ax.add_patch(Rectangle((lx, 160), 26, 18, fc=col, ec="none", alpha=a, zorder=6))
        txt(ax, lx + 36, 163, name, 18, INK2, a)
        lx += 60 + 11 * len(name)
    if part == 0:
        txt(ax, W - 140, 163, "median gain over the 14 test images: +0.2 F1", 16, MUTED, a, ha="right")
    caption(ax, cap, a * seg(tt, 1.0, 1.6))


SCENES = [(s_title, 7.0), (s_problem, 9.0), (s_recipe, 11.0), (s_patches, 12.5), (s_embed, 11.5),
          (s_contrast, 15.0), (s_consistency, 10.5), (s_objective, 10.0), (s_fair, 10.0), (s_res_monuseg, 11.5),
          (s_res_chaos, 11.5), (s_real, 17.0)] + ([(s_learned, 12.0)] if EMB else []) + [(s_ablation, 10.0),
                                                                                      (s_outro, 9.0)]
BUSIEST = [4.5, 7.5, 9.5, 8.0, 9.5, 12.0, 9.0, 8.0, 9.5, 10.5, 10.0, 6.0] + ([11.0] if EMB else []) + [9.5, 8.5]
TOTAL = sum(d for _, d in SCENES)


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

    def frame(self, si, t):
        fn, D = SCENES[si]
        ax = self.ax
        ax.cla()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), W, H, fc=BG, ec="none", zorder=-10))
        fn(ax, t, D)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        gt = sum(d for _, d in SCENES[:si]) + t
        ax.add_patch(Rectangle((0, 0), W * gt / TOTAL, 5, fc=BLUE, ec="none", zorder=20))
        if si > 0:
            txt(ax, W - 110, 975, "%02d / %02d" % (si, len(SCENES) - 1), 16, MUTED, 1, ha="right", z=20)
        fade = min(1.0, t / 0.4, (D - t) / 0.4)
        if fade < 1:
            ax.add_patch(Rectangle((0, 0), W, H, fc=BG, ec="none", alpha=1 - ease(fade), zorder=30))
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def render_scene(cv, si, out):
    D = SCENES[si][1]
    tmp = out[:-4] + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-r", str(FPS), tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(round(D * FPS))):
        p.stdin.write(cv.frame(si, (i + 0.5) / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed on scene %d" % si)
    os.replace(tmp, out)


def ambient(path, dur, sr=44100):
    t = np.arange(int(dur * sr)) / sr
    chords = [[220.0, 261.63, 329.63, 392.0], [174.61, 220.0, 261.63, 329.63], [130.81, 196.0, 261.63, 329.63],
              [196.0, 246.94, 293.66, 392.0]]
    out = np.zeros((len(t), 2))
    L = 8.0
    for c in range(int(dur // L) + 2):
        t0 = c * L - 1.0
        env = np.clip((t - t0) / 2.5, 0, 1) * np.clip((t0 + L + 2.0 - t) / 2.5, 0, 1)
        if not env.any():
            continue
        for j, f in enumerate(chords[c % 4]):
            ph = 2 * np.pi * f * t
            v = (np.sin(ph) + 0.25 * np.sin(2 * ph + 0.3) + 0.08 * np.sin(3 * ph)) * env
            pan = 0.35 + 0.1 * j
            lfo = 1 + 0.15 * np.sin(2 * np.pi * (0.07 + 0.02 * j) * t)
            out[:, 0] += v * lfo * (1 - pan)
            out[:, 1] += v * lfo * pan
    out *= (np.clip(t / 2.0, 0, 1) * np.clip((dur - t) / 3.0, 0, 1))[:, None]
    out = out / (np.abs(out).max() + 1e-9) * 10 ** (-20 / 20)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "cssl_explainer.mp4"))
    a = ap.parse_args()
    cv = Canvas()
    if a.stills:
        from PIL import Image
        os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
        for si in (a.only if a.only is not None else range(len(SCENES))):
            Image.fromarray(cv.frame(si, BUSIEST[si])).save(os.path.join(HERE, "stills", "s%02d.png" % si))
        print("stills in", os.path.join(HERE, "stills"))
        return
    build = os.path.join(HERE, "build")
    os.makedirs(build, exist_ok=True)
    for si in (a.only if a.only is not None else range(len(SCENES))):
        render_scene(cv, si, os.path.join(build, "scene%02d.mp4" % si))
        print("scene %02d done" % si, flush=True)
    lst = os.path.join(build, "list.txt")
    with open(lst, "w") as f:
        for si in range(len(SCENES)):
            f.write("file 'scene%02d.mp4'\n" % si)
    silent = os.path.join(build, "silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    silent], check=True)
    wav = os.path.join(build, "pad.wav")
    ambient(wav, TOTAL)
    tmp = a.out[:-4] + ".part.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", tmp],
                   check=True)
    os.replace(tmp, a.out)
    print("video:", a.out, "(%.1f s)" % TOTAL)


if __name__ == "__main__":
    main()
