#!/usr/bin/env python3

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cssl import AdaptivePerturbation, cssl_contrastive_loss, sample_class_patches, silhouette_l1
from cssl import selfloop as SL
from cssl.supcon import gaussian_aux_negatives
from data.datasets import ChaosSlices, ChaosVolumes, Mixture, MonusegCrops, MonusegTest, load_split
from model.cssl_net import CSSLNet
from util.unimatch_utils import DiceLoss

CLASS_NAMES = {"chaos": ["liver", "right_kidney", "left_kidney", "spleen"], "monuseg": ["nucleus"]}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--label-frac", required=True, help="chaos: 50 | 25 | 13   monuseg: 20 | 50 | 100 (all labeled)")
    p.add_argument("--save-path", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", help="checkpoint to evaluate on the test set")
    p.add_argument("--opts", nargs="*", default=[], help="overrides, e.g. cssl.cl.weight=0.05 total_iters=10000")
    p.add_argument("--bn-recalib", type=int, default=0,
                   help="with --eval-only: re-estimate the BN running statistics on this many batches of ordinary "
                        "training images before testing (diagnostic; the result is kept under *_bnrecal keys)")
    return p.parse_args()


def apply_overrides(cfg, opts):
    for o in opts:
        key, val = o.split("=", 1)
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def sigmoid_rampup(it, length):
    if length <= 0 or it >= length:
        return 1.0
    return float(math.exp(-5.0 * (1.0 - max(it, 0) / float(length)) ** 2))


@contextmanager
def bn_stats_frozen(model):
    saved = [(m, m.momentum) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m, _ in saved:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom in saved:
            m.momentum = mom


def infinite(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def recalibrate_bn(net, set_l, set_u, batch_size, n_batches, device, num_workers=2):
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.eval()
    for m in bns:
        m.train()
    mk = lambda ds: infinite(DataLoader(ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True))
    il, iu = mk(set_l), mk(set_u)
    for _ in range(n_batches):
        net(torch.cat([next(il)[0], next(iu)[0]]).to(device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    net.eval()


def build_data(cfg, split, frac):
    root, size = cfg["data_root"], cfg["crop_size"]
    if cfg["dataset"] == "chaos":
        labeled = split["labeled"][frac]
        unlabeled = [s for s in split["train"] if s not in labeled]
        geom = cfg.get("geom_aug", "unimatch")
        ctx, aug, aff = int(cfg.get("chaos_context", 0)), cfg.get("chaos_aug", "unimatch"), cfg.get("affine")
        if cfg["in_chns"] != 2 * ctx + 1:
            raise ValueError("in_chns must be 2 * chaos_context + 1 = %d" % (2 * ctx + 1))
        kw = dict(geom=geom, context=ctx, aug=aug, affine=aff)
        if cfg.get("chaos_protocol", "t2") == "dafnet":
            t2 = lambda ids: ["%s_t2" % s for s in ids]
            set_u = ChaosSlices(root, t2(unlabeled), "train_u", size, **kw)
            parts = [(ChaosSlices(root, t2(labeled), "train_l", size, **kw), float(cfg.get("t2_share", 0.5)))]
            if cfg.get("use_t1_labels", True):
                parts.append((ChaosSlices(root, ["%s_t1" % s for s in split["train"]], "train_l", size, **kw),
                              1.0 - float(cfg.get("t2_share", 0.5))))
            set_l = Mixture(parts, max(len(set_u), sum(len(p[0]) for p in parts)))
            return (set_l, set_u, ChaosVolumes(root, t2(split["val"]), ctx), ChaosVolumes(root, t2(split["test"]), ctx),
                    labeled)
        set_u = ChaosSlices(root, unlabeled, "train_u", size, **kw)
        set_l = ChaosSlices(root, labeled, "train_l", size, nsample=len(set_u), **kw)
        return (set_l, set_u, ChaosVolumes(root, split["val"], ctx), ChaosVolumes(root, split["test"], ctx),
                labeled)
    labeled = list(split["train"]) if frac == "100" else split["labeled"][frac]
    val = []
    k = int(cfg.get("val_labeled", 0))
    if k and frac != "100":
        labeled, val = list(labeled[:-k]), list(labeled[-k:])
    unlabeled = [n for n in split["train"] if n not in labeled and n not in val] or list(labeled)
    cpi = cfg.get("crops_per_image", 64)
    sa = cfg.get("strong_aug", "unimatch")
    scale = tuple(cfg.get("scale_aug", (1.0, 1.0)))
    set_l = MonusegCrops(root, labeled, "train_l", size, crops_per_image=cpi * max(1, len(unlabeled) // len(labeled)),
                         scale=scale, labeled_color=cfg.get("labeled_color", "none"))
    set_u = MonusegCrops(root, unlabeled, "train_u", size, crops_per_image=cpi, strong_aug=sa, scale=scale)
    return set_l, set_u, (MonusegTest(root, val, split="train") if val else None), MonusegTest(root, split["test"]), labeled


def largest_component_per_class(pred, nclass):
    from scipy import ndimage
    p = pred.cpu().numpy()
    out = np.zeros_like(p)
    for c in range(1, nclass + 1):
        lab, n = ndimage.label(p == c)
        if n:
            sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
            out[lab == (int(np.argmax(sizes)) + 1)] = c
    return torch.from_numpy(out).to(pred.device)


def dafnet_dice_masks(P, gt, nclass):
    G = F.one_hot(gt.long(), nclass)[..., 1:].float()
    P = P.float()
    inter, den = (P * G).sum((1, 2, 3)), P.sum((1, 2, 3)) + G.sum((1, 2, 3))
    pooled = float(((2 * inter + 1e-12) / (den + 1e-12)).mean())
    ic, dc = (P * G).sum((1, 2)), P.sum((1, 2)) + G.sum((1, 2))
    return pooled, ((2 * ic + 1e-12) / (dc + 1e-12)).mean(0).tolist()


def dafnet_dice(pred, gt, nclass):
    P = F.one_hot(pred.long(), nclass)[..., 1:].float()
    G = F.one_hot(gt.long(), nclass)[..., 1:].float()
    inter, den = (P * G).sum((1, 2, 3)), P.sum((1, 2, 3)) + G.sum((1, 2, 3))
    pooled = float(((2 * inter + 1e-12) / (den + 1e-12)).mean())
    ic, dc = (P * G).sum((1, 2)), P.sum((1, 2)) + G.sum((1, 2))
    organs = ((2 * ic + 1e-12) / (dc + 1e-12)).mean(0).tolist()
    return pooled, organs


def pooled_binary_scores(tp, fp, fn, tn):
    f1_fg = 2 * tp / max(1, 2 * tp + fp + fn)
    f1_bg = 2 * tn / max(1, 2 * tn + fn + fp)
    bacc = (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2
    return f1_fg, (f1_fg + f1_bg) / 2, bacc


def dice_score(pred, gt, c):
    p, g = pred == c, gt == c
    s = int(p.sum()) + int(g.sum())
    return 1.0 if s == 0 else 2.0 * int((p & g).sum()) / s


@torch.no_grad()
def evaluate(model, dataset, cfg, device, tta=None, lcc=None):
    tta = cfg.get("eval_tta", False) if tta is None else tta
    lcc = cfg.get("eval_lcc", False) if lcc is None else lcc
    was_training = model.training
    model.eval()
    names = CLASS_NAMES[cfg["dataset"]]
    per_case, fgbg, bacc, daf, daf_round = [], [], [], [], []
    conf = [0, 0, 0, 0]
    for x, gt, cid in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
        gt = gt[0].to(device)
        if cfg["dataset"] == "chaos":
            vol = x[0].to(device)
            h, w = vol.shape[-2:]
            preds, rounded = [], []
            for k in range(0, vol.shape[0], 16):
                v = F.interpolate(vol[k:k + 16], (cfg["crop_size"], cfg["crop_size"]), mode="bilinear", align_corners=False)
                out = F.interpolate(model(v).float(), (h, w), mode="bilinear", align_corners=False)
                preds.append(out.argmax(1))
                rounded.append((out.softmax(1)[:, 1:] > 0.5).permute(0, 2, 3, 1))
            pred = torch.cat(preds)
            daf_round.append(dafnet_dice_masks(torch.cat(rounded), gt, len(names) + 1))
            if lcc:
                pred = largest_component_per_class(pred, len(names))
        else:
            img = x.to(device)
            h, w = img.shape[-2:]
            ph, pw = (-h) % 32, (-w) % 32
            img = F.pad(img, (0, pw, 0, ph), mode="reflect")
            if tta:
                prob = 0
                for k in range(4):
                    for flip in (False, True):
                        t = torch.rot90(img, k, (2, 3))
                        t = torch.flip(t, (3,)) if flip else t
                        o = model(t).float().softmax(1)
                        o = torch.flip(o, (3,)) if flip else o
                        prob = prob + torch.rot90(o, -k, (2, 3))
                pred = prob[..., :h, :w].argmax(1)[0]
            else:
                pred = model(img).float()[..., :h, :w].argmax(1)[0]
        per_case.append([dice_score(pred, gt, c) for c in range(1, len(names) + 1)])
        if cfg["dataset"] == "chaos":
            daf.append(dafnet_dice(pred, gt, len(names) + 1))
        if cfg["dataset"] == "monuseg":
            fgbg.append((dice_score(pred, gt, 1) + dice_score(pred, gt, 0)) / 2)
            rec = [int(((pred == c) & (gt == c)).sum()) / max(1, int((gt == c).sum())) for c in (0, 1)]
            bacc.append(sum(rec) / 2)
            p1, g1 = pred == 1, gt == 1
            for i, v in enumerate((p1 & g1, p1 & ~g1, ~p1 & g1, ~p1 & ~g1)):
                conf[i] += int(v.sum())
    model.train(was_training)
    arr = np.array(per_case) * 100
    per_class = arr.mean(0)
    extra = {}
    if fgbg:
        pd_, pf, pb = pooled_binary_scores(*conf)
        extra = {"selfloop_f1_fg_bg_mean": float(np.mean(fgbg) * 100), "balanced_accuracy": float(np.mean(bacc) * 100),
                 "per_case_fg_bg_mean": (np.array(fgbg) * 100).tolist(),
                 "pooled_dice_nucleus": pd_ * 100, "pooled_f1_fg_bg_mean": pf * 100, "pooled_balanced_accuracy": pb * 100}
    if daf:
        pooled = np.array([d[0] for d in daf]) * 100
        organs = np.array([d[1] for d in daf]) * 100
        if cfg.get("chaos_metric", "organ3d") == "dafnet":
            extra = {"dice_3d_organ_mean": float(per_class.mean()),
                     "dice_3d_per_class": dict(zip(names, per_class.tolist())), "per_case_3d": arr.tolist(),
                     "dafnet_dice_round": float(np.mean([d[0] for d in daf_round]) * 100)}
            return (float(pooled.mean()), dict(zip(names, organs.mean(0).tolist())),
                    np.concatenate([pooled[:, None], organs], 1).tolist(), extra)
        extra = {"dafnet_dice": float(pooled.mean()), "dafnet_dice_per_class": dict(zip(names, organs.mean(0).tolist()))}
    if daf_round:
        extra["dafnet_dice_round"] = float(np.mean([d[0] for d in daf_round]) * 100)
    return float(per_class.mean()), dict(zip(names, per_class.tolist())), arr.tolist(), extra


def main():
    args = parse_args()
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.opts)
    frac = str(args.label_frac)
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger("cssl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.save_path, "train.log"))):
        h.setFormatter(fmt)
        logger.addHandler(h)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    gen_cpu = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    split_dir = cfg.get("split_dir") or os.path.join(HERE, "splits")
    if not os.path.isabs(split_dir):
        split_dir = os.path.join(HERE, split_dir)
    split = load_split(split_dir, cfg["dataset"], args.split_seed, cfg.get("split_prefix", "seed"))
    set_l, set_u, set_val, set_test, labeled = build_data(cfg, split, frac)
    logger.info("config:\n%s", yaml.safe_dump(cfg, sort_keys=False))
    logger.info("dataset %s, split %s%d, %s%% labeled = %d cases (%s), %d labeled / %d unlabeled samples",
                cfg["dataset"], cfg.get("split_prefix", "seed"), args.split_seed, frac, len(labeled),
                ",".join(map(str, labeled)), len(set_l), len(set_u))
    if isinstance(set_l, Mixture):
        logger.info("labeled batches: %s", ", ".join("%d slices (%s...) at share %.2f" % (len(d), d.items[0][0], sh)
                                                    for d, sh in set_l.parts))
    if not args.eval_only:
        yaml.safe_dump({**cfg, "_args": vars(args)}, open(os.path.join(args.save_path, "config.yaml"), "w"), sort_keys=False)

    nclass = cfg["nclass"]
    cl_cfg, cr_cfg = cfg["cssl"]["cl"], cfg["cssl"]["cr"]
    method = cfg.get("method", "cssl")
    if method not in ("cssl", "selfloop"):
        raise ValueError("method must be cssl or selfloop")
    sl_cfg = cfg.get("selfloop") or {}
    model = CSSLNet(cfg["in_chns"], nclass, tuple(cfg["widths"]), cl_cfg["embed_dim"],
                    backbone=cfg.get("backbone", "unet"), pretrained=cfg.get("pretrained", True),
                    input_norm=cfg.get("input_norm", "auto"),
                    jigsaw_classes=int(sl_cfg.get("K", 100)) if method == "selfloop" else 0,
                    jigsaw_hidden=int(sl_cfg.get("head_hidden", 512)),
                    embed_level=int(cl_cfg.get("embed_level", -1))).to(device)
    bb = cfg.get("backbone", "unet")
    seg_params = sum(p.numel() for p in model.unet.parameters()) / 1e6
    network = "%s, %.1fM, %s" % ("U-Net %s" % "-".join(map(str, cfg["widths"])) if bb == "unet" else
                                 bb.replace("resunet", "ResUNet-"), seg_params,
                                 "ImageNet" if (bb != "unet" and cfg.get("pretrained", True)) else "random init")
    logger.info("model: %s (with projection head %.2fM parameters)", network,
                sum(p.numel() for p in model.parameters()) / 1e6)
    enc = list(model.unet.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": enc, "lr_mult": float(cfg.get("encoder_lr_mult", 1.0))}, {"params": rest, "lr_mult": 1.0}]
    opt_name = cfg.get("optimizer", "sgd")
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg.get("weight_decay", 0.01))
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(groups, lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
    else:
        raise ValueError("optimizer must be sgd or adamw")
    if cr_cfg.get("fixed_lambda") is None:
        perturber = AdaptivePerturbation(cr_cfg["epsilon"], cr_cfg["omega_momentum"], cr_cfg.get("relative", True))
    else:
        perturber = AdaptivePerturbation(cr_cfg["epsilon"], 1.0, cr_cfg.get("relative", True),
                                         init_omega=2.0 * (float(cr_cfg["fixed_lambda"]) - cr_cfg["epsilon"]) - 1.0)
    aux = gaussian_aux_negatives(cl_cfg["embed_dim"], cl_cfg.get("aux_negatives", 256), seed=args.seed, device=device)
    cl_classes = list(range(1, nclass)) + ([0] if cl_cfg.get("include_background", False) else [])
    bnd_w = int(cl_cfg.get("boundary", 0))
    if bnd_w:
        if nclass != 2:
            raise ValueError("cssl.cl.boundary is for binary segmentation")
        cl_classes = cl_classes + [nclass]
    q_size = int(cl_cfg.get("queue", 0))
    mem_z = torch.zeros(0, cl_cfg["embed_dim"], device=device)
    mem_l = torch.zeros(0, dtype=torch.long, device=device)

    ema_cfg = cfg.get("ema") or {}
    use_ema = bool(ema_cfg.get("enable", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    teacher = None
    if use_ema:
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

    @torch.no_grad()
    def ema_update(it):
        d = min(ema_decay, (1.0 + it) / (10.0 + it))
        tp, sp = list(teacher.parameters()), [p.detach() for p in model.parameters()]
        torch._foreach_mul_(tp, d)
        torch._foreach_add_(tp, sp, alpha=1.0 - d)
        for bt, bs in zip(teacher.buffers(), model.buffers()):
            if bt.dtype.is_floating_point:
                bt.mul_(d).add_(bs, alpha=1.0 - d)
            else:
                bt.copy_(bs)

    eval_net = teacher if use_ema else model
    select_best = cfg.get("select_best", False)
    best = {"val": -1.0, "iter": -1}

    enc_opt, pool, sl_q, sl_th = None, None, int(sl_cfg.get("Q", 10)), float(sl_cfg.get("th", 0.95))
    if method == "selfloop":
        pool = torch.from_numpy(SL.permutation_pool(int(sl_cfg.get("K", 100)), seed=args.seed))
        enc_params = list(model.unet.encoder.parameters()) + list(model.jigsaw.parameters())
        enc_opt = torch.optim.SGD([{"params": enc_params, "lr_mult": float(cfg.get("encoder_lr_mult", 1.0))}],
                                  lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
        dist = (pool[:, None] != pool[None]).sum(-1).float()
        logger.info("Self-Loop: K=%d permutations (mean pairwise Hamming distance %.2f), Q=%d, th=%.2f, jigsaw BN %s",
                    len(pool), float(dist[torch.triu(torch.ones_like(dist, dtype=torch.bool), 1)].mean()), sl_q, sl_th,
                    sl_cfg.get("aux_bn", "frozen"))
    sl_bn = (lambda: bn_stats_frozen(model)) if sl_cfg.get("aux_bn", "frozen") == "frozen" else contextlib.nullcontext
    if sl_cfg.get("aux_bn", "frozen") not in ("frozen", "shared"):
        raise ValueError("selfloop.aux_bn must be frozen or shared")

    def save(path, it):
        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "perturb": perturber.state_dict(),
              "iter": it, "cfg": cfg, "best": dict(best)}
        if use_ema:
            ck["ema"] = teacher.state_dict()
        if enc_opt is not None:
            ck["enc_opt"] = enc_opt.state_dict()
        ck["mem_z"], ck["mem_l"] = mem_z.detach().cpu(), mem_l.cpu()
        ck["rng"] = {"gen": gen.get_state(), "gen_cpu": gen_cpu.get_state(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "numpy": np.random.get_state(), "random": random.getstate()}
        torch.save(ck, path)

    def load_eval_weights(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        eval_net.load_state_dict(ck["ema"] if (use_ema and "ema" in ck) else ck["model"])
        return ck

    def full_test(net, tag):
        m, pc, cases, extra = evaluate(net, set_test, cfg, device, tta=False, lcc=False)
        out = {"test_mean_dice": m, "per_class": pc, "per_case": cases, **extra}
        if cfg["dataset"] == "monuseg":
            m2, pc2, _, e2 = evaluate(net, set_test, cfg, device, tta=True, lcc=False)
            out.update({"test_mean_dice_tta": m2, "per_class_tta": pc2, **{k + "_tta": v for k, v in e2.items()}})
        else:
            m2, pc2, _, _ = evaluate(net, set_test, cfg, device, tta=False, lcc=True)
            out.update({"test_mean_dice_lcc": m2, "per_class_lcc": pc2})
        short = {k: round(v, 2) for k, v in out.items() if isinstance(v, float)}
        logger.info("FINAL test (%s) mean Dice %.2f %s %s", tag, m, {k: round(v, 2) for k, v in pc.items()}, short)
        return out

    if args.eval_only:
        load_eval_weights(args.eval_only)
        if args.bn_recalib:
            m, pc, _, extra = evaluate(eval_net, set_test, cfg, device)
            logger.info("TEST before BN re-estimation: mean Dice %.2f %s", m,
                        {k: round(v, 2) for k, v in extra.items() if isinstance(v, float)})
            recalibrate_bn(eval_net, set_l, set_u, cfg["batch_size"], args.bn_recalib, device,
                           cfg.get("num_workers", 2))
            logger.info("BN running statistics re-estimated on %d batches of ordinary training crops", args.bn_recalib)
        m, pc, cases, extra = evaluate(eval_net, set_test, cfg, device)
        logger.info("TEST%s mean Dice %.2f %s %s", " after BN re-estimation:" if args.bn_recalib else "", m,
                    {k: round(v, 2) for k, v in pc.items()},
                    {k: round(v, 2) for k, v in extra.items() if isinstance(v, float)})
        res_path = os.path.join(args.save_path, "results.json")
        suffix = "_tta" if cfg.get("eval_tta") else "_lcc" if cfg.get("eval_lcc") else ""
        suffix += "_bnrecal" if args.bn_recalib else ""
        if os.path.exists(res_path):
            res = json.load(open(res_path))
            if not suffix and abs(res.get("test_mean_dice", m) - m) > 0.05:
                logger.info("note: re-evaluated Dice %.2f differs from results.json %.2f", m, res["test_mean_dice"])
            if suffix:
                extra = {"test_mean_dice" + suffix: m, "per_class" + suffix: pc,
                         **{k + suffix: v for k, v in extra.items()}}
            res.update(extra)
            json.dump(res, open(res_path, "w"), indent=2)
            logger.info("added %s to %s", ", ".join(k for k in extra if not k.startswith("per_case")) or "nothing", res_path)
        return

    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)
    pin = device.type == "cuda"
    mk = lambda ds: DataLoader(ds, bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True,
                               persistent_workers=nw > 0)
    it_l, it_u, it_m = infinite(mk(set_l)), infinite(mk(set_u)), infinite(mk(set_u))
    ce = nn.CrossEntropyLoss()
    ce_u = nn.CrossEntropyLoss(reduction="none")
    dice = DiceLoss(nclass)
    total, warm = cfg["total_iters"], cfg["warmup_iters"]
    lr_warm = int(cfg.get("lr_warmup_iters", 0))
    thr = cfg["conf_thresh"]
    mk_amp = lambda: (torch.autocast("cuda", dtype=torch.bfloat16) if (cfg.get("amp") == "bf16" and pin)
                      else torch.autocast(device.type, enabled=False))
    amp = mk_amp()

    start = 0
    latest = os.path.join(args.save_path, "latest.pth")
    best_path = os.path.join(args.save_path, "best.pth")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if len(ck["optimizer"]["param_groups"]) == 1:
            optimizer = torch.optim.SGD([{"params": list(model.parameters()), "lr_mult": 1.0}], lr=cfg["lr"],
                                        momentum=0.9, weight_decay=cfg.get("weight_decay", 1e-4))
        optimizer.load_state_dict(ck["optimizer"])
        perturber.load_state_dict(ck["perturb"])
        if enc_opt is not None and "enc_opt" in ck:
            enc_opt.load_state_dict(ck["enc_opt"])
        if use_ema:
            teacher.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        best.update(ck.get("best", {}))
        if "mem_z" in ck:
            mem_z, mem_l = ck["mem_z"].to(device), ck["mem_l"].to(device)
        if "rng" in ck:
            r = ck["rng"]
            gen.set_state(r["gen"])
            gen_cpu.set_state(r["gen_cpu"])
            torch.set_rng_state(r["torch"])
            if r["cuda"] is not None and device.type == "cuda":
                torch.cuda.set_rng_state_all(r["cuda"])
            np.random.set_state(r["numpy"])
            random.setstate(r["random"])
        start = ck["iter"] + 1
        logger.info("resumed at iteration %d (memory bank %d entries)", start, mem_z.shape[0])

    def validate(it):
        m, pc, _, _ = evaluate(eval_net, set_val, cfg, device, tta=False, lcc=False)
        logger.info("val (it %d%s) mean Dice %.2f %s", it, ", EMA teacher" if use_ema else "", m,
                    {k: round(v, 1) for k, v in pc.items()})
        metrics_f.write(json.dumps({"iter": it, "val_dice": m}) + "\n")
        if select_best and m > best["val"]:
            best.update(val=m, iter=it)
            save(best_path, it - 1)
            logger.info("best validation Dice so far: %.2f at it %d (saved best.pth)", m, it)

    metrics_f = open(os.path.join(args.save_path, "metrics.jsonl"), "a")
    meter, t_last, it_last = {}, time.time(), start
    n_bad = 0
    for it in range(start, total):
        semi = it >= warm
        lr = cfg["lr"] * (1 - it / total) ** 0.9
        if it < lr_warm:
            lr *= (it + 1) / lr_warm
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        model.train()
        x, y = (t.to(device, non_blocking=True) for t in next(it_l))
        logs = {}
        if method == "selfloop" and semi:
            u_sl = next(it_u)[0].to(device, non_blocking=True)
            for g in enc_opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            with sl_bn():
                y_sl, l_unl = SL.self_loop_pseudo_label(model, u_sl, pool, sl_q, enc_opt, gen_cpu, mk_amp)
            model.train()
        with amp:
            if method == "selfloop":
                pred_x = model(x).float()
                loss_x = (ce(pred_x, y) + dice(pred_x.softmax(1), y.unsqueeze(1).float())) / 2.0
                with sl_bn():
                    loss_ss, jig_acc = SL.jigsaw_loss(model, x, pool, sl_q, gen_cpu)
                loss = loss_x + loss_ss
                logs.update(loss_ss_lab=float(loss_ss.detach()), jigsaw_acc=float(jig_acc))
                if semi:
                    pred_u = model(u_sl).float().softmax(1)
                    loss_ug, ug_px = SL.uncertainty_guided_loss(pred_u, y_sl, sl_th)
                    loss = loss + loss_ug
                    logs.update(loss_ug=float(loss_ug.detach()), ug_pixels=ug_px, loss_ss_unl=l_unl)
            elif not semi:
                pred_x = model(x).float()
                loss_x = (ce(pred_x, y) + dice(pred_x.softmax(1), y.unsqueeze(1).float())) / 2.0
                loss = loss_x
            else:
                u_w, s1, s2, b1, b2 = (t.to(device, non_blocking=True) for t in next(it_u))
                u_w_m, s1_m, s2_m, _, _ = (t.to(device, non_blocking=True) for t in next(it_m))
                nu = u_w.shape[0]
                with torch.no_grad():
                    if use_ema:
                        conf_t, mask_t = teacher(torch.cat([u_w, u_w_m])).float().softmax(1).max(1)
                        conf_w, conf_m = conf_t.split([nu, u_w_m.shape[0]])
                        mask_w, mask_m = mask_t.split([nu, u_w_m.shape[0]])
                    else:
                        model.eval()
                        conf_m, mask_m = model(u_w_m).float().softmax(1).max(1)
                        model.train()
                s1 = torch.where(b1[:, None].bool(), s1_m, s1)
                s2 = torch.where(b2[:, None].bool(), s2_m, s2)
                nl = x.shape[0]
                if cr_cfg.get("enable", True):
                    preds, preds_fp = model(torch.cat([x, u_w]), "seg_cr", perturb=perturber)
                elif cr_cfg.get("fallback", "dropout") == "none":
                    preds, preds_fp = model(torch.cat([x, u_w])), None
                else:
                    preds, preds_fp = model(torch.cat([x, u_w]), "seg_fp")
                pred_x, pred_w = preds.float().split([nl, nu])
                ps1, ps2 = model(torch.cat([s1, s2])).float().chunk(2)
                if not use_ema:
                    conf_w, mask_w = pred_w.detach().softmax(1).max(1)
                mask1, conf1 = torch.where(b1.bool(), mask_m, mask_w), torch.where(b1.bool(), conf_m, conf_w)
                mask2, conf2 = torch.where(b2.bool(), mask_m, mask_w), torch.where(b2.bool(), conf_m, conf_w)

                loss_x = (ce(pred_x, y) + dice(pred_x.softmax(1), y.unsqueeze(1).float())) / 2.0
                loss_s1 = dice(ps1.softmax(1), mask1.unsqueeze(1).float(), ignore=(conf1 < thr).float())
                loss_s2 = dice(ps2.softmax(1), mask2.unsqueeze(1).float(), ignore=(conf2 < thr).float())
                keep = (conf_w >= thr).float()
                if preds_fp is None:
                    loss_cr = loss_x.new_zeros(())
                    loss = (loss_x + 0.5 * loss_s1 + 0.5 * loss_s2) / 2.0
                else:
                    pred_fp = preds_fp.float()[nl:]
                    loss_cr = ((ce_u(pred_fp, mask_w) * keep).mean() +
                               dice(pred_fp.softmax(1), mask_w.unsqueeze(1).float(), ignore=1 - keep)) / 2.0
                    if cr_cfg.get("mode", "replace") == "add":
                        loss = (loss_x + 0.5 * loss_s1 + 0.5 * loss_s2 + float(cr_cfg.get("add_weight", 0.5)) * loss_cr) / 2.0
                    else:
                        loss = (loss_x + 0.25 * loss_s1 + 0.25 * loss_s2 + 0.5 * loss_cr) / 2.0
                logs.update(loss_s=float((loss_s1 + loss_s2).detach()) / 2, loss_cr=float(loss_cr.detach()),
                            mask_ratio=float(keep.mean()), lam=perturber.lam)

                if cl_cfg.get("enable", True):
                    pmap = mask_w.clone()
                    pmap[conf_w < thr] = 255
                    lab_cl = torch.cat([y, pmap])
                    if bnd_w:
                        near_bg = F.max_pool2d((lab_cl == 0).float()[:, None], 2 * bnd_w + 1, 1, bnd_w)[:, 0] > 0
                        lab_cl = torch.where((lab_cl == 1) & near_bg, torch.full_like(lab_cl, nclass), lab_cl)
                    P = sample_class_patches(torch.cat([x, u_w]).float(), lab_cl, cl_classes,
                                             cl_cfg["patch_size"], cl_cfg["stride"], cl_cfg["min_pixels"],
                                             cl_cfg["max_patches"], generator=gen_cpu,
                                             mode=cl_cfg.get("patch_mode", "prominent"))
                    q, n_p = int(cl_cfg.get("patch_quantum", 1)), P["labels"].numel()
                    if q > 1 and n_p > q and n_p % q:
                        sel = torch.randperm(n_p, generator=gen_cpu)[:n_p - n_p % q].to(device)
                        P = {k: v[sel] for k, v in P.items()}
                    loss_cl = torch.zeros((), device=device)
                    if P["labels"].numel() > 1:
                        with bn_stats_frozen(model):
                            if cl_cfg.get("embed_from", "patch") == "image":
                                z = model(torch.cat([x, u_w]), "embed_image", masks=P["masks"], boxes=P["boxes"])
                            else:
                                z = model(P["patches"], "embed", masks=P["masks"])
                        one_class = P["labels"].unique().numel() < 2
                        loss_cl, info = cssl_contrastive_loss(
                            z, P["labels"], tau=cl_cfg["tau"], hard_ratio=cl_cfg["hard_ratio"],
                            alpha_max=cl_cfg["alpha_max"], max_hard=cl_cfg["max_hard"],
                            aux_negatives=aux if one_class else None, generator=gen,
                            memory=(mem_z, mem_l) if q_size else None)
                        if q_size:
                            mem_z = torch.cat([z.detach().float(), mem_z])[:q_size]
                            mem_l = torch.cat([P["labels"], mem_l])[:q_size]
                        omega = silhouette_l1(z.detach(), P["labels"], generator=gen_cpu)
                        perturber.update(omega)
                        logs.update(n_patches=int(P["labels"].numel()), cl_anchors=info["cl_anchors"],
                                    omega=omega if omega is not None else float("nan"))
                    w_cl = cl_cfg["weight"] * sigmoid_rampup(it - warm, cl_cfg["rampup_iters"])
                    if not bool(torch.isfinite(loss_cl)):
                        loss_cl = loss_cl.new_zeros(())
                    loss = loss + w_cl * loss_cl
                    logs.update(loss_cl=float(loss_cl.detach()), w_cl=w_cl)

        optimizer.zero_grad(set_to_none=True)
        if bool(torch.isfinite(loss)):
            loss.backward()
            optimizer.step()
            if use_ema:
                ema_update(it)
        else:
            n_bad += 1
            logger.info("warning: non-finite loss at it %d, step skipped (%d so far)", it + 1, n_bad)
            if n_bad > 100:
                raise RuntimeError("more than 100 non-finite losses: stopping")

        logs.update(loss=float(loss.detach()), loss_x=float(loss_x.detach()))
        for k, v in logs.items():
            meter.setdefault(k, []).append(v)
        if (it + 1) % cfg.get("log_every", 50) == 0 or it == start:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                avg = {k: float(np.nanmean(v)) for k, v in meter.items()}
            meter = {}
            now = time.time()
            spi = (now - t_last) / max(1, it + 1 - it_last)
            t_last, it_last = now, it + 1
            mem = torch.cuda.max_memory_allocated() / 1e9 if pin else 0.0
            logger.info("it %d/%d %s lr %.5f | %s | %.3fs/it eta %.2fh max_mem %.1fGB", it + 1, total,
                        "semi" if semi else "warm", lr, " ".join("%s %.3f" % kv for kv in avg.items()),
                        spi, spi * (total - it - 1) / 3600, mem)
            metrics_f.write(json.dumps({"iter": it + 1, **avg}) + "\n")
            metrics_f.flush()
        if (it + 1) % cfg["eval_every"] == 0 and (it + 1) < total:
            if set_val is not None:
                validate(it + 1)
            save(latest, it)

    save(os.path.join(args.save_path, "final.pth"), total - 1)
    if set_val is not None:
        validate(total)
    save(latest, total - 1)
    if cfg.get("no_test", False):
        m, pc, _, extra = evaluate(eval_net, set_val, cfg, device, tta=False, lcc=False)
        res = {"dataset": cfg["dataset"], "label_frac": frac, "split_seed": args.split_seed, "labeled_cases": labeled,
               "val_cases": list(getattr(set_val, "names", [])), "iters": total, "network": network, "val_dice": m,
               **{"val_" + k: v for k, v in extra.items() if not k.startswith("per_case")}}
        json.dump(res, open(os.path.join(args.save_path, "results.json"), "w"), indent=2)
        logger.info("VALIDATION (model selection; test set not evaluated) nuclei F1 %.2f %s", m,
                    {k: round(v, 2) for k, v in extra.items() if isinstance(v, float)})
        return
    res = {"dataset": cfg["dataset"], "label_frac": frac, "split_seed": args.split_seed, "labeled_cases": labeled,
           "iters": total, "evaluated": "EMA teacher" if use_ema else "student", "network": network,
           "input_slices": 2 * int(cfg.get("chaos_context", 0)) + 1 if cfg["dataset"] == "chaos" else 1}
    final_out = full_test(eval_net, "final weights" + (", EMA teacher" if use_ema else ""))
    if select_best and best["iter"] > 0:
        if best["iter"] != total:
            load_eval_weights(best_path)
        sel_out = full_test(eval_net, "best validation Dice %.2f at it %d" % (best["val"], best["iter"]))
        if select_best == "report":
            res.update(final_out, selected="final weights", best_val_dice=best["val"], best_iter=best["iter"],
                       best_val_checkpoint={k: v for k, v in sel_out.items() if not k.startswith("per_case")})
        else:
            res.update(sel_out, selected="best validation checkpoint", best_val_dice=best["val"],
                       best_iter=best["iter"],
                       final_weights={k: v for k, v in final_out.items() if not k.startswith("per_case")})
    else:
        res.update(final_out, selected="final weights")
    if use_ema:
        m, pc, _, extra = evaluate(model, set_test, cfg, device, tta=False, lcc=False)
        res["student_final"] = {"test_mean_dice": m, "per_class": pc,
                                **{k: v for k, v in extra.items() if not k.startswith("per_case")}}
        logger.info("FINAL test (student, final weights) mean Dice %.2f %s", m, {k: round(v, 2) for k, v in pc.items()})
    json.dump(res, open(os.path.join(args.save_path, "results.json"), "w"), indent=2)
    logger.info("results: %s", os.path.join(args.save_path, "results.json"))


if __name__ == "__main__":
    main()
