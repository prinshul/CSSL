#!/usr/bin/env python3

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools.make_splits import DAFNET_SPLITS, MONUSEG_TRAIN_30

PROBLEMS = []


def check(ok, msg):
    print(("  OK       " if ok else "  PROBLEM  ") + msg)
    if not ok:
        PROBLEMS.append(msg)


def pixel_hash(path):
    from PIL import Image
    return hashlib.md5(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def monuseg(data, splits):
    print("\n[1] MoNuSeg: split and leakage")
    root = os.path.join(data, "monuseg")
    for k in range(3):
        s = load_json(os.path.join(splits, "monuseg", "seed%d.json" % k))
        if s is None:
            if k == 0:
                check(False, "splits/monuseg/seed0.json not found")
            continue
        tr, te, lab = set(s["train"]), set(s["test"]), s["labeled"]
        tag = "seed%d:" % k
        check(len(tr) == 30 and len(te) == 14, "%s %d training / %d test images (expect 30 / 14)" % (tag, len(tr), len(te)))
        check(not (tr & te), "%s no image name in both training and test" % tag)
        check(all(any(n.startswith(c) for c in MONUSEG_TRAIN_30) for n in tr),
              "%s every training image is one of the 30 original Kumar et al. images" % tag)
        for f, n in (("20", 6), ("50", 15)):
            check(len(lab[f]) == n and set(lab[f]) <= tr,
                  "%s %s%% labeled = %d images, all from the training set (expect %d)" % (tag, f, len(lab[f]), n))
    s = load_json(os.path.join(splits, "monuseg", "seed0.json"))
    if s is None or not os.path.isdir(root):
        return s
    h = {}
    for part, names in (("train", s["train"]), ("test", s["test"])):
        for n in names:
            p = os.path.join(root, part, "images", n + ".png")
            if os.path.exists(p):
                h.setdefault(pixel_hash(p), []).append("%s/%s" % (part, n))
            else:
                check(False, "missing image %s" % p)
    dup = [v for v in h.values() if len(v) > 1]
    check(not dup, "no two images share identical pixels across training and test%s" % (": %s" % dup if dup else ""))
    return s


def chaos(data, splits):
    print("\n[2] CHAOS (DAFNet protocol): splits and leakage")
    out = {}
    for k in range(3):
        s = load_json(os.path.join(splits, "chaos", "dafnet%d.json" % k))
        if s is None:
            check(False, "splits/chaos/dafnet%d.json not found" % k)
            continue
        out[k] = s
        tr, va, te = set(map(str, s["train"])), set(map(str, s["val"])), set(map(str, s["test"]))
        lab = set(map(str, s["labeled"]["25"]))
        d = DAFNET_SPLITS[k]
        tag = "split %d:" % k
        check(tr == set(map(str, d["training"])) and va == set(map(str, d["validation"])) and te == set(map(str, d["test"])),
              "%s train / val / test subjects are DAFNet's own" % tag)
        check(not (tr & va) and not (tr & te) and not (va & te), "%s train, val and test subjects are disjoint" % tag)
        check(len(lab) == 4 and lab <= tr, "%s 25%% labeled = %d T2 subjects (%s), all training subjects (expect 4)"
              % (tag, len(lab), ",".join(sorted(lab, key=int))))
    root = os.path.join(data, "chaos_dafnet")
    if os.path.isdir(root) and out:
        h = {}
        for f in sorted(glob.glob(os.path.join(root, "*_t[12].npz"))):
            z = np.load(f)
            key = [k for k in z.files if k not in ("label", "mask", "labels", "masks")][0]
            h.setdefault(hashlib.md5(np.ascontiguousarray(z[key]).tobytes()).hexdigest(), []).append(os.path.basename(f))
        dup = [v for v in h.values() if len(v) > 1]
        check(not dup, "no two CHAOS volumes are identical (%d volumes)%s" % (len(h), (": %s" % dup) if dup else ""))
    return out


def runs(exp, mon_split, chaos_splits, splits_dir):
    print("\n[3] Finished runs: labeled set used, network, weights tested")
    rows = []
    for f in sorted(glob.glob(os.path.join(exp, "*", "*", "results.json"))):
        r = json.load(open(f))
        d = os.path.dirname(f)
        tag = os.path.relpath(d, exp)
        try:
            import yaml
            cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        except Exception:
            cfg = {}
        lab = list(map(str, r.get("labeled_cases", [])))
        if r["dataset"] == "monuseg" and mon_split:
            k = int(r.get("split_seed", 0))
            sp = mon_split if k == 0 else (load_json(os.path.join(splits_dir, "monuseg", "seed%d.json" % k)) or {})
            exp_lab = (sp.get("labeled") or {}).get(str(r["label_frac"]), sp.get("train", []))
            check(sorted(lab) == sorted(map(str, exp_lab)),
                  "%-34s labeled images = split seed%d's %s%% list (%d)" % (tag, k, r["label_frac"], len(lab)))
        elif r["dataset"] == "chaos" and r.get("split_seed") in chaos_splits:
            exp_lab = chaos_splits[r["split_seed"]]["labeled"][str(r["label_frac"])]
            check(sorted(lab) == sorted(map(str, exp_lab)),
                  "%-34s labeled T2 subjects = DAFNet split %d's (%s)" % (tag, r["split_seed"], ",".join(lab)))
        log = os.path.join(d, "train.log")
        res = [l.strip() for l in open(log, errors="ignore") if "resumed at iteration" in l] if os.path.exists(log) else []
        if not res:
            check(True, "%-34s trained in one go (no resume)" % tag)
        else:
            full = all("memory bank" in l for l in res)
            check(full, "%-34s resumed %d time(s)%s" % (tag, len(res), ", memory bank and random state restored" if full
                                                         else " by older code: memory bank and random state not restored"))
        rows.append((tag, r, cfg))
    nets = {}
    for tag, r, cfg in rows:
        nets.setdefault(r["dataset"], set()).add(r.get("network", "?"))
    for ds, n in nets.items():
        check(len(n) == 1, "%s: every run uses the same network: %s" % (ds, " | ".join(sorted(n))))
    for tag, r, cfg in rows:
        sel = cfg.get("select_best", False) if cfg else "?"
        how = "final weights" if sel in (False, "report") else "best validation checkpoint (validation subjects only)"
        print("  info     %-34s %s, tested: %s" % (tag, cfg.get("method", "cssl") if cfg else "?", how))
    return rows


def activity(exp, rows):
    print("\n[4] Were the CSSL components active? (semi-supervised phase, from metrics.jsonl)")
    for tag, r, cfg in rows:
        if not cfg or cfg.get("method", "cssl") != "cssl":
            continue
        m = os.path.join(exp, tag, "metrics.jsonl")
        if not os.path.exists(m):
            continue
        recs = [x for x in map(json.loads, open(m)) if "loss_x" in x and ("lam" in x or "loss_s" in x)]
        if not recs:
            print("  info     %-34s supervised only (no semi-supervised phase)" % tag)
            continue
        cl_on = cfg["cssl"]["cl"].get("enable", True)
        cr_on = cfg["cssl"]["cr"].get("enable", True)
        fixed = cfg["cssl"]["cr"].get("fixed_lambda") is not None
        get = lambda k: np.array([x[k] for x in recs if k in x and isinstance(x[k], (int, float)) and math.isfinite(x[k])])
        lam, om, npat, anc, lcl, cr = get("lam"), get("omega"), get("n_patches"), get("cl_anchors"), get("loss_cl"), get("loss_cr")
        q = max(1, len(lcl) // 4)
        desc = "L_CL %s, L_CR %s%s" % ("on" if cl_on else "off", "on" if cr_on else "off", ", lambda fixed" if fixed else "")
        print("  info     %-34s %s" % (tag, desc))
        if cl_on:
            check(len(npat) and npat.mean() > 1 and len(anc) and anc.mean() > 0,
                  "%-34s contrastive loss had patches and anchors: %.0f patches, %.0f anchors per step"
                  % (tag, npat.mean() if len(npat) else 0, anc.mean() if len(anc) else 0))
            if len(lcl) > 4:
                print("  info     %-34s L_CL first quarter %.3f -> last quarter %.3f" % (tag, lcl[:q].mean(), lcl[-q:].mean()))
            if len(om):
                print("  info     %-34s Silhouette omega: start %.3f, mean %.3f, end %.3f" % (tag, om[:q].mean(), om.mean(), om[-q:].mean()))
        if cr_on and len(lam):
            spread = lam.max() - lam.min()
            if fixed:
                check(spread < 1e-6, "%-34s lambda held constant at %.4f" % (tag, lam.mean()))
            elif cl_on:
                check(spread > 1e-3, "%-34s lambda followed the Silhouette score: %.3f -> %.3f (min %.3f, max %.3f)"
                      % (tag, lam[:q].mean(), lam[-q:].mean(), lam.min(), lam.max()))
            if len(cr):
                mr = get("mask_ratio")
                check(cr.mean() > 0, "%-34s consistency loss active: mean L_CR %.4f on %.0f%% confident pixels"
                      % (tag, cr.mean(), 100 * (mr.mean() if len(mr) else 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with monuseg/ and chaos_dafnet/")
    p.add_argument("--exp", default=os.path.join(HERE, "exp"))
    p.add_argument("--splits", default=os.path.join(HERE, "splits"))
    a = p.parse_args()
    ms = monuseg(a.data, a.splits)
    cs = chaos(a.data, a.splits)
    rows = runs(a.exp, ms, cs, a.splits)
    activity(a.exp, rows)
    print("\n%d problem(s) found" % len(PROBLEMS))
    for m in PROBLEMS:
        print("  - " + m)


if __name__ == "__main__":
    main()
