import argparse
import glob
import json
import os

import numpy as np

SELFLOOP, DAFNET = 79.10, 82.0


def load(d):
    f = os.path.join(d, "results.json")
    return json.load(open(f)) if os.path.exists(f) else None


def one(d):
    r = load(d)
    if r is None:
        return ""
    if "test_mean_dice" not in r and "val_dice" in r:
        return "val %.2f" % r["val_dice"]
    s = "test %.2f" % r["test_mean_dice"]
    return s


def score(E, tag, key, k, final=False, metric="test_mean_dice"):
    r = load(os.path.join(E, tag, "%s_s%d" % (key, k)))
    if r is None or metric not in r:
        return None
    if final:
        return (r.get("final_weights") or {}).get(metric, r[metric])
    return r[metric]


def fmt(v):
    return "%7.2f" % v if v is not None else "%7s" % "-"


def row(name, vals, ref=None):
    done = [v for v in vals if v is not None]
    m = ("%6.2f +- %4.2f" % (np.mean(done), np.std(done))) if done else "%15s" % "-"
    extra = ""
    if ref is not None and len(done) == len(vals) and all(x is not None for x in ref):
        d = np.array(vals) - np.array(ref)
        extra = "   vs recipe %+.2f, higher on %d/%d" % (d.mean(), int((d > 0).sum()), len(d))
    print("  %-30s %s   %s%s" % (name, " ".join(fmt(v) for v in vals), m, extra))


def table(E, title, key, rows, ref_tag, published, final=False, metric="test_mean_dice"):
    print("\n%s" % title)
    print("  %-30s %7s %7s %7s   %s" % ("", "s0", "s1", "s2", "mean +- sd"))
    ref = [score(E, ref_tag, key, k, final, metric) for k in range(3)]
    for name, tag in rows:
        vals = [score(E, tag, key, k, final, metric) for k in range(3)]
        row(name, vals, ref if tag == "cssl_strongest" else None)
    print("  %-30s %31.2f" % published)


def ablation(E, title, key, k, rows):
    print("\n%s" % title)
    full = score(E, "cssl_strongest", key, k)
    for name, tag in rows:
        v = score(E, tag, key, k)
        if v is None:
            continue
        d = "" if tag == "cssl_strongest" or full is None else "   full - this %+.2f" % (full - v)
        print("  %-30s %7.2f%s" % (name, v, d))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp")
    ap.add_argument("--one", default=None)
    a = ap.parse_args()
    if a.one:
        print(one(a.one))
        return
    E = os.path.abspath(a.exp)
    print("=" * 96)
    print("RESULTS (test sets)")
    print("=" * 96)
    table(E, "MoNuSeg 50% labels, nuclei F1, final weights", "monuseg_50",
          [("Supervised", "main_sup"), ("Recipe only", "mabl_none"),
           ("CSSL", "cssl_strongest")], "mabl_none", ("Self-Loop (published)", SELFLOOP))
    table(E, "CHAOS T2 25% labels, DAFNet Dice, best-validation checkpoint", "chaos_25",
          [("Supervised", "main_sup"), ("Recipe only", "cabl_none"), ("CSSL", "cssl_strongest")], "cabl_none",
          ("DAFNet (published)", DAFNET))
    table(E, "CHAOS T2 25% labels, same checkpoints, DAFNet's rounding (each organ probability > 0.5)", "chaos_25",
          [("Supervised", "main_sup"), ("Recipe only", "cabl_none"), ("CSSL", "cssl_strongest")], "cabl_none",
          ("DAFNet (published)", DAFNET), metric="dafnet_dice_round")
    abl = [("Full CSSL", "cssl_strongest"), ("w/o contrastive loss", "sabl_no_cl"), ("w/o consistency loss", "sabl_no_cr"),
           ("w/o hard negatives", "sabl_no_hnm"), ("w/o Silhouette (fixed lambda)", "sabl_fixlam"),
           ("w/o boundary class", "sabl_no_bnd"), ("w/o feature-level masking", "sabl_no_v2"),
           ("w/o memory bank", "sabl_no_mem")]
    ablation(E, "Ablation, MoNuSeg seed 0", "monuseg_50", 0, abl + [("Recipe only", "mabl_none")])
    ablation(E, "Ablation, CHAOS split 1", "chaos_25", 1, abl + [("Recipe only", "cabl_none")])
    print("=" * 96)


if __name__ == "__main__":
    main()
