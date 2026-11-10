#!/usr/bin/env python3

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
N = json.load(open(os.path.join(HERE, "numbers.json")))

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID, MID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f0efec"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 9, "axes.titlecolor": INK,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def rounded_bar(ax, x, h, base, w, color, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, base), w, h - base, boxstyle="round,pad=0,rounding_size=0.03",
                                linewidth=0, facecolor=color, mutation_aspect=(h - base) / max(w, 1e-6) * 0.05, **kw))


def fig_results():
    from matplotlib.lines import Line2D
    m, c = N["monuseg"], N["chaos"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.26})
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]

    def mean_tick(ax, x, mu, col, half=0.2):
        ax.plot([x - half, x + half], [mu, mu], color=col, lw=2.4, solid_capstyle="round", zorder=4)

    groups = [("Supervised\n(1 run)", [m["supervised_50_s0"]["nuclei_f1"]], AQUA),
              ("Recipe\nonly", m["recipe"], ORANGE),
              ("CSSL\n(ours)", m["cssl_full"], BLUE)]
    sl = m["selfloop_published"]
    a1.axhline(sl, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    a1.text(-0.42, sl + 0.1, f"Self-Loop (published) {sl:.2f}", ha="left", va="bottom", fontsize=7.0, color=INK,
            path_effects=halo, zorder=6)
    for i, (name, vals, col) in enumerate(groups):
        mu = float(np.mean(vals))
        xs = np.full(len(vals), i) + (np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else 0)
        a1.scatter(xs, vals, s=30, color=col, edgecolor="white", linewidth=1.0, zorder=5)
        if len(vals) > 1:
            mean_tick(a1, i, mu, col)
        sd = f"\n± {np.std(vals):.2f}" if len(vals) > 1 else ""
        a1.text(i + 0.24, mu, f"{mu:.2f}{sd}", ha="left", va="center", color=INK, fontsize=7.3,
                fontweight="bold" if i == 2 else "normal", linespacing=1.05, path_effects=halo, zorder=6)
    a1.set_xticks(range(3), [g[0] for g in groups])
    a1.set_xlim(-0.45, 2.75)
    a1.set_ylim(75.0, 82.0)
    a1.set_ylabel("Nuclei F1 (%)")
    a1.set_title("MoNuSeg, 50% labels (3 seeds)", loc="left")
    a1.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    a1.set_axisbelow(True)

    rows = [("Supervised", c["supervised"]["best_val"], AQUA), ("Recipe only", c["recipe"]["best_val"], ORANGE),
            ("CSSL (ours)", c["cssl_full"]["best_val"], BLUE)]
    cats = ["Split 0", "Split 1", "Split 2", "Mean"]
    d = c["dafnet_published"]
    a2.axhline(d, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    a2.axvspan(2.55, 3.62, color=MID, zorder=0, lw=0)
    for j, (name, vals, col) in enumerate(rows):
        ys = list(vals) + [float(np.mean(vals))]
        xs = np.arange(4) + (j - 1) * 0.24
        a2.scatter(xs[:3], ys[:3], s=30, color=col, edgecolor="white", linewidth=1.0, zorder=5, label=name)
        a2.scatter(xs[3:], ys[3:], s=62, color=col, edgecolor="white", linewidth=1.2, marker="D", zorder=5)
        off = {0: (0.0, -0.75, "center", "top"), 1: (-0.13, 0.0, "right", "center"), 2: (0.13, 0.0, "left", "center")}[j]
        a2.text(xs[3] + off[0], ys[3] + off[1], f"{ys[3]:.1f}", ha=off[2], va=off[3], fontsize=7.0, color=INK,
                fontweight="bold" if j == 2 else "normal", path_effects=halo, zorder=6)
    a2.set_xticks(range(4), cats)
    a2.set_xlim(-0.5, 3.62)
    a2.set_ylim(76.0, 97.0)
    a2.set_yticks([80, 85, 90, 95])
    a2.set_ylabel("Dice (%), DAFNet's protocol")
    a2.set_title("CHAOS, 25% labels (DAFNet's 3 splits)", loc="left")
    a2.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    a2.set_axisbelow(True)
    hs, ls = a2.get_legend_handles_labels()
    hs.append(Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2))))
    ls.append(f"DAFNet (published) {d:.1f}")
    a2.legend(hs, ls, loc="upper left", ncol=4, fontsize=6.6, handlelength=1.4, columnspacing=0.8,
              handletextpad=0.3, bbox_to_anchor=(0.0, 1.01), borderaxespad=0.2)
    save(fig, "fig_results")


def fig_ablation():
    m, c = N["monuseg"]["ablation_s0"], N["chaos"]["ablation_s1"]
    panels = [("MoNuSeg (seed 0); full CSSL = %.2f" % m["Full"], m,
               ["Recipe only", "w/o boundary class", "w/o CL", "w/o feature-level embedding", "w/o HNM", "w/o CR",
                "w/o Silhouette", "w/o memory bank"]),
              ("CHAOS (split 1); full CSSL = %.2f" % c["Full"], c,
               ["Recipe only", "w/o CL", "w/o Silhouette", "w/o HNM", "w/o CR"])]
    nice = {"Recipe only": "all of CSSL (recipe only)", "w/o boundary class": "boundary class",
            "w/o CL": "contrastive loss (CL)", "w/o feature-level embedding": "feature-level embedding",
            "w/o HNM": "hard negatives (HNM)", "w/o CR": "consistency reg. (CR)",
            "w/o Silhouette": "Silhouette-adaptive λ", "w/o memory bank": "memory bank"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.95})
    for ax, (title, tab, order) in zip(axes, panels):
        deltas = [tab["Full"] - tab[k] for k in order]
        ys = np.arange(len(order))[::-1]
        for y, dlt in zip(ys, deltas):
            col = BLUE if dlt > 0 else RED
            ax.barh(y, dlt, height=0.62, color=col, edgecolor="white", linewidth=1.0, zorder=2)
            ax.text(max(dlt, 0) + 0.12, y, f"{dlt:+.2f}".replace("-", "\u2212"), va="center", ha="left",
                    fontsize=7.2, color=INK)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=3)
        ax.set_yticks(ys, [nice[k] for k in order], fontsize=7.4)
        lo = min(0, min(deltas)) - 1.0
        hi = max(deltas) + 1.4
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Drop in score when removed (points)")
        ax.set_title(title, loc="left", fontsize=8.4)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    save(fig, "fig_ablation")


def fig_calibration():
    m = N["monuseg"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
    readings = ["Nuclei F1", "Mean of nuclei and\nbackground F1", "Balanced accuracy"]
    for ax, frac, ref in zip(axes, ["50", "100"], [m["selfloop_published_sup50"], m["selfloop_published_sup100"]]):
        ours = m["supervised_%s_s0" % frac]
        vals = [ours["nuclei_f1"], ours["fgbg_f1"], ours["bal_acc"]]
        ys = np.arange(3)[::-1]
        ax.axvline(ref, color=INK, lw=1.0, ls=(0, (4, 2)))
        ax.text(ref, 3.05, f"Self-Loop's supervised: {ref:.2f}", ha="center", va="bottom", fontsize=7.2, color=INK,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        for y, v in zip(ys, vals):
            ax.plot([ref, v], [y, y], color=GRID, lw=2.0, zorder=1)
            ax.scatter([v], [y], s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(v + (0.35 if v >= ref else -0.35), y, f"{v:.2f} ({v - ref:+.2f})", va="center",
                    ha="left" if v >= ref else "right", fontsize=7.2, color=INK)
        ax.set_yticks(ys, readings, fontsize=7.4)
        ax.set_xlim(72.5, 92.5)
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(f"Supervised ResUNet-18, {frac}% labels", loc="left", fontsize=8.2)
        ax.set_xlabel("Our supervised score under each reading (%)")
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    save(fig, "fig_calibration")


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=7.2, bold=False, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, color=MUTED, lw=0.9, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=1, shrinkB=1))


def synthetic_tissue(n=96, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    mask = np.zeros((n, n), bool)
    for _ in range(9):
        cy, cx = rng.uniform(8, n - 8, 2)
        ry, rx = rng.uniform(5, 9, 2)
        th = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = (dx * np.cos(th) + dy * np.sin(th)) / rx
        v = (-dx * np.sin(th) + dy * np.cos(th)) / ry
        mask |= u ** 2 + v ** 2 <= 1
    noise = rng.normal(0, 0.035, (n, n, 1))
    pink = np.array([0.93, 0.72, 0.80])
    purple = np.array([0.40, 0.24, 0.56])
    img = np.where(mask[..., None], purple, pink) + noise
    return np.clip(img, 0, 1), mask


def fig_method():
    fig = plt.figure(figsize=(7.0, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    GFC, GEC = "#f3f2ee", "#9a9893"
    BFC, BEC = "#e6f0fb", BLUE

    ax.text(0.01, 0.995, "Recipe (shared by every semi-supervised run)", fontsize=7.6, color=INK2, va="top")
    ax.text(0.53, 0.995, "CSSL components (the only difference to the recipe)", fontsize=7.6, color=BLUE, va="top",
            fontweight="bold")
    box(ax, 0.01, 0.76, 0.12, 0.13, "Labeled\nimage $x_l$,\nlabel $y$", "white", GEC)
    box(ax, 0.01, 0.36, 0.12, 0.13, "Unlabeled\nimage $x_u$", "white", GEC)
    box(ax, 0.17, 0.76, 0.13, 0.13, "Student\n$f_\\theta$ = encoder $h$\n+ decoder $g$", GFC, GEC)
    box(ax, 0.34, 0.785, 0.15, 0.08, "$\\mathcal{L}_{sup}$: CE + Dice", GFC, GEC)
    box(ax, 0.17, 0.50, 0.13, 0.11, "Weak view\n→ EMA teacher $f_\\xi$", GFC, GEC)
    box(ax, 0.34, 0.505, 0.15, 0.10, "Pseudo-label $\\hat y$\n(confidence ≥ 0.95)", GFC, GEC)
    box(ax, 0.17, 0.20, 0.13, 0.12, "2 strong views\n+ CutMix\n→ student", GFC, GEC)
    box(ax, 0.34, 0.215, 0.15, 0.09, "$\\mathcal{L}_{s_1}, \\mathcal{L}_{s_2}$:\nweak-to-strong", GFC, GEC)
    arrow(ax, (0.13, 0.825), (0.17, 0.825))
    arrow(ax, (0.30, 0.825), (0.34, 0.825))
    arrow(ax, (0.13, 0.43), (0.17, 0.555), rad=-0.15)
    arrow(ax, (0.13, 0.40), (0.17, 0.26), rad=0.15)
    arrow(ax, (0.30, 0.555), (0.34, 0.555))
    arrow(ax, (0.415, 0.505), (0.415, 0.305))
    arrow(ax, (0.30, 0.26), (0.34, 0.26))
    box(ax, 0.53, 0.74, 0.18, 0.15, "Class-wise patches\nfrom $y$ and $\\hat y$\n(nucleus, boundary,\nbackground)", BFC, BEC)
    box(ax, 0.75, 0.74, 0.22, 0.15, "Feature-level masked pooling\nof encoder features (1/8)\n→ projection head → $z$", BFC, BEC)
    box(ax, 0.75, 0.47, 0.22, 0.17, "$\\mathcal{L}_{CL}$: supervised contrastive\nloss + mixed hard negatives\n+ memory bank (4096)", BFC, BEC)
    box(ax, 0.53, 0.47, 0.18, 0.17, "Silhouette $\\omega$ (L1) of\nclass clusters in $z$\n$\\lambda = (\\omega + 1)/2 + \\epsilon$", BFC, BEC)
    box(ax, 0.53, 0.19, 0.18, 0.18, "Encoder features $h$\n+ noise $\\lambda\\,\\sigma(h)\\,\\xi$\n→ decoder", BFC, BEC)
    box(ax, 0.75, 0.20, 0.22, 0.16, "$\\mathcal{L}_{CR}$: perturbed prediction\nmatches $\\hat y$ (CE + Dice),\nadded on top of the recipe", BFC, BEC)
    ax.plot([0.07, 0.07, 0.60], [0.89, 0.935, 0.935], color=BLUE, lw=0.9)
    arrow(ax, (0.60, 0.935), (0.60, 0.89), color=BLUE)
    ax.text(0.33, 0.94, "labels $y$", fontsize=6.8, color=BLUE, ha="center", va="bottom")
    arrow(ax, (0.49, 0.585), (0.58, 0.74), color=BLUE, rad=-0.2)
    arrow(ax, (0.71, 0.815), (0.75, 0.815), color=BLUE)
    arrow(ax, (0.86, 0.74), (0.86, 0.64), color=BLUE)
    arrow(ax, (0.75, 0.555), (0.71, 0.555), color=BLUE)
    arrow(ax, (0.62, 0.47), (0.62, 0.37), color=BLUE)
    arrow(ax, (0.71, 0.28), (0.75, 0.28), color=BLUE)
    ax.text(0.5, 0.075, "$\\mathcal{L} = \\frac{1}{2}\\left(\\mathcal{L}_{sup} + \\frac{1}{2}\\mathcal{L}_{s_1} + "
            "\\frac{1}{2}\\mathcal{L}_{s_2} + \\beta\\,\\mathcal{L}_{CR}\\right) + w\\,\\mathcal{L}_{CL}$",
            ha="center", va="center", fontsize=10, color=INK)
    ax.text(0.5, 0.025, "gray: the recipe, kept at full strength in every run    blue: CSSL's components, added on top",
            ha="center", va="center", fontsize=7.0, color=INK2)
    save(fig, "fig_method")


def fig_embedding():
    img, mask = synthetic_tissue()
    n = img.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), gridspec_kw={"wspace": 0.1, "width_ratios": [1, 2.06, 1, 1]})
    titles = ["(a) A 64×64 window\n", "(b) Pixel-level masking\n",
              "(c) Feature-level\nmasking", "(d) Boundary class\n"]
    axes[0].imshow(img, interpolation="nearest")
    nuc = img * mask[..., None]
    bg = img * (~mask)[..., None]
    both = np.concatenate([nuc, np.ones((n, 4, 3)), bg], 1)
    axes[1].imshow(both, interpolation="nearest")
    axes[1].text(n / 2, n + 9, "nucleus patch", ha="center", fontsize=6.8, color=INK2)
    axes[1].text(n + 4 + n / 2, n + 9, "background patch", ha="center", fontsize=6.8, color=INK2)
    axes[2].imshow(img, interpolation="nearest")
    cs = n // 8
    for k in range(9):
        axes[2].axhline(k * cs - 0.5, color="white", lw=0.5, alpha=0.9)
        axes[2].axvline(k * cs - 0.5, color="white", lw=0.5, alpha=0.9)
    for i in range(8):
        for j in range(8):
            f = mask[i * cs:(i + 1) * cs, j * cs:(j + 1) * cs].mean()
            if f > 0.02:
                axes[2].add_patch(Rectangle((j * cs - 0.5, i * cs - 0.5), cs, cs, fc=BLUE, alpha=0.18 + 0.55 * f, lw=0))
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(mask, iterations=2)
    rim = mask & ~inner
    lab = np.zeros((n, n, 3)) + np.array([0.96, 0.95, 0.93])
    lab[inner] = np.array([0.40, 0.24, 0.56])
    lab[rim] = np.array([0.92, 0.41, 0.20])
    axes[3].imshow(lab, interpolation="nearest")
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=7.6, loc="center", fontweight="normal", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].set_ylim(n + 14, -0.5)
    save(fig, "fig_embedding")


if __name__ == "__main__":
    fig_results()
    fig_ablation()
    fig_calibration()
    fig_method()
    fig_embedding()
    print("figures written to", OUT)
