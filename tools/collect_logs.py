#!/usr/bin/env python3

import argparse
import glob
import hashlib
import json
import os
import shutil

import yaml

MON, CHA = "monuseg_50", "chaos_25"
ITEMS = [
    ("Table 1", "MoNuSeg supervised 50%, seed 0", "main_sup", MON, 0, "test_mean_dice", 76.01),
    ("Table 1", "MoNuSeg supervised 100%, seed 0", "SUP100", "monuseg_100", 0, "test_mean_dice", 77.64),
    ("Table 1", "MoNuSeg recipe only, seed 0", "mabl_none", MON, 0, "test_mean_dice", 77.69),
    ("Table 1", "MoNuSeg recipe only, seed 1", "mabl_none", MON, 1, "test_mean_dice", 78.50),
    ("Table 1", "MoNuSeg recipe only, seed 2", "mabl_none", MON, 2, "test_mean_dice", 79.99),
    ("Table 1", "MoNuSeg CSSL, seed 0", "cssl_strongest", MON, 0, "test_mean_dice", 80.66),
    ("Table 1", "MoNuSeg CSSL, seed 1", "cssl_strongest", MON, 1, "test_mean_dice", 81.07),
    ("Table 1", "MoNuSeg CSSL, seed 2", "cssl_strongest", MON, 2, "test_mean_dice", 80.88),
    ("F1 calibration", "supervised 50%: fg/bg F1", "main_sup", MON, 0, "selfloop_f1_fg_bg_mean", 84.41),
    ("F1 calibration", "supervised 50%: balanced accuracy", "main_sup", MON, 0, "balanced_accuracy", 85.23),
    ("F1 calibration", "supervised 100%: fg/bg F1", "SUP100", "monuseg_100", 0, "selfloop_f1_fg_bg_mean", 84.62),
    ("F1 calibration", "supervised 100%: balanced accuracy", "SUP100", "monuseg_100", 0, "balanced_accuracy", 87.49),
    ("Appendix B", "CSSL seed 0: fg/bg F1", "cssl_strongest", MON, 0, "selfloop_f1_fg_bg_mean", 87.73),
    ("Appendix B", "CSSL seed 0: balanced accuracy", "cssl_strongest", MON, 0, "balanced_accuracy", 87.27),
    ("Appendix B", "CSSL seed 0: pooled-pixel nuclei F1", "cssl_strongest", MON, 0, "pooled_dice_nucleus", 81.00),
    ("Table 2", "CHAOS supervised, split 0", "main_sup", CHA, 0, "test_mean_dice", 90.55),
    ("Table 2", "CHAOS supervised, split 1", "main_sup", CHA, 1, "test_mean_dice", 79.07),
    ("Table 2", "CHAOS supervised, split 2", "main_sup", CHA, 2, "test_mean_dice", 87.98),
    ("Table 2", "CHAOS recipe only, split 0", "cabl_none", CHA, 0, "test_mean_dice", 89.96),
    ("Table 2", "CHAOS recipe only, split 1", "cabl_none", CHA, 1, "test_mean_dice", 85.33),
    ("Table 2", "CHAOS recipe only, split 2", "cabl_none", CHA, 2, "test_mean_dice", 91.52),
    ("Table 2", "CHAOS CSSL, split 0", "cssl_strongest", CHA, 0, "test_mean_dice", 90.47),
    ("Table 2", "CHAOS CSSL, split 1", "cssl_strongest", CHA, 1, "test_mean_dice", 85.67),
    ("Table 2", "CHAOS CSSL, split 2", "cssl_strongest", CHA, 2, "test_mean_dice", 90.90),
    ("Table 5", "CHAOS supervised, split 0, rounding", "main_sup", CHA, 0, "dafnet_dice_round", 90.60),
    ("Table 5", "CHAOS supervised, split 1, rounding", "main_sup", CHA, 1, "dafnet_dice_round", 79.02),
    ("Table 5", "CHAOS supervised, split 2, rounding", "main_sup", CHA, 2, "dafnet_dice_round", 88.03),
    ("Table 5", "CHAOS recipe only, split 0, rounding", "cabl_none", CHA, 0, "dafnet_dice_round", 90.02),
    ("Table 5", "CHAOS recipe only, split 1, rounding", "cabl_none", CHA, 1, "dafnet_dice_round", 85.32),
    ("Table 5", "CHAOS recipe only, split 2, rounding", "cabl_none", CHA, 2, "dafnet_dice_round", 91.54),
    ("Table 5", "CHAOS CSSL, split 0, rounding", "cssl_strongest", CHA, 0, "dafnet_dice_round", 90.55),
    ("Table 5", "CHAOS CSSL, split 1, rounding", "cssl_strongest", CHA, 1, "dafnet_dice_round", 85.66),
    ("Table 5", "CHAOS CSSL, split 2, rounding", "cssl_strongest", CHA, 2, "dafnet_dice_round", 90.93),
    ("Table 3", "MoNuSeg w/o contrastive loss", "sabl_no_cl", MON, 0, "test_mean_dice", 78.51),
    ("Table 3", "MoNuSeg w/o consistency loss", "sabl_no_cr", MON, 0, "test_mean_dice", 80.42),
    ("Table 3", "MoNuSeg w/o hard negatives", "sabl_no_hnm", MON, 0, "test_mean_dice", 80.07),
    ("Table 3", "MoNuSeg w/o Silhouette (fixed lambda)", "sabl_fixlam", MON, 0, "test_mean_dice", 80.54),
    ("Table 3", "MoNuSeg w/o memory bank", "sabl_no_mem", MON, 0, "test_mean_dice", 81.14),
    ("Table 3", "MoNuSeg w/o boundary class", "sabl_no_bnd", MON, 0, "test_mean_dice", 77.99),
    ("Table 3", "MoNuSeg w/o feature-level masking", "sabl_no_v2", MON, 0, "test_mean_dice", 79.11),
    ("Table 3", "CHAOS w/o contrastive loss", "sabl_no_cl", CHA, 1, "test_mean_dice", 78.46),
    ("Table 3", "CHAOS w/o consistency loss", "sabl_no_cr", CHA, 1, "test_mean_dice", 86.50),
    ("Table 3", "CHAOS w/o hard negatives", "sabl_no_hnm", CHA, 1, "test_mean_dice", 85.16),
    ("Table 3", "CHAOS w/o Silhouette (fixed lambda)", "sabl_fixlam", CHA, 1, "test_mean_dice", 84.99),
]
FILES = ("config.yaml", "train.log", "metrics.jsonl", "results.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_sup100(exp):
    for d in sorted(glob.glob(os.path.join(exp, "*", "monuseg_100_s0"))):
        c = os.path.join(d, "config.yaml")
        if os.path.exists(os.path.join(d, "results.json")) and os.path.exists(c):
            cfg = yaml.safe_load(open(c))
            if int(cfg.get("warmup_iters", 0)) >= int(cfg.get("total_iters", 1)):
                return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="logs")
    ap.add_argument("--hash-weights", action="store_true")
    a = ap.parse_args()
    exp = os.path.abspath(a.exp)
    os.makedirs(a.out, exist_ok=True)
    sup100 = find_sup100(exp)
    rows, runs, ok = [], {}, 0
    for table, what, tag, key, k, metric, paper in ITEMS:
        d = sup100 if tag == "SUP100" else os.path.join(exp, tag, "%s_s%d" % (key, k))
        rel = os.path.relpath(d, exp) if d else "not found"
        val = None
        if d and os.path.exists(os.path.join(d, "results.json")):
            val = json.load(open(os.path.join(d, "results.json"))).get(metric)
            runs[rel] = d
        match = val is not None and abs(round(float(val) + 1e-12, 2) - paper) < 0.005
        ok += match
        rows.append((table, what, rel, metric, "-" if val is None else "%.4f" % val, "%.2f" % paper,
                     "yes" if match else "NO"))
        print("%-15s %-42s %-34s %-24s %10s  paper %6s  %s" % rows[-1])
    for rel, d in sorted(runs.items()):
        dst = os.path.join(a.out, "runs", rel)
        os.makedirs(dst, exist_ok=True)
        for f in FILES:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), dst)
    with open(os.path.join(a.out, "INDEX.md"), "w") as f:
        f.write("# Logs of the reported runs\n\n")
        f.write("Every number of the paper's tables, the run folder it comes from and the value in that run's "
                "`results.json`. Each run folder under `runs/` holds its `config.yaml`, `train.log`, "
                "`metrics.jsonl` and `results.json`.\n\n")
        f.write("%d of %d paper values match their run (to 0.005).\n\n" % (ok, len(ITEMS)))
        f.write("| Paper | Item | Run | Key in results.json | Value | Paper | Match |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %s |\n" % " | ".join(r))
        f.write("\n## Files\n\n| File | SHA-256 |\n|---|---|\n")
        for rel, d in sorted(runs.items()):
            for fn in FILES:
                p = os.path.join(a.out, "runs", rel, fn)
                if os.path.exists(p):
                    f.write("| runs/%s/%s | `%s` |\n" % (rel, fn, sha256(p)))
            if a.hash_weights:
                for w in ("final.pth", "best.pth"):
                    p = os.path.join(d, w)
                    if os.path.exists(p):
                        f.write("| %s/%s (not included) | `%s` |\n" % (rel, w, sha256(p)))
    print("\n%d of %d paper values match their run; %d runs copied to %s/runs, index in %s/INDEX.md"
          % (ok, len(ITEMS), len(runs), a.out, a.out))


if __name__ == "__main__":
    main()
