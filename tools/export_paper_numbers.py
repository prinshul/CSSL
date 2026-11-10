#!/usr/bin/env python3

import argparse
import copy
import glob
import json
import os
import sys

import yaml


def load(exp, tag, key, k):
    f = os.path.join(exp, tag, "%s_s%d" % (key, k), "results.json")
    return json.load(open(f)) if os.path.exists(f) else None


def r2(v):
    return None if v is None else round(float(v) + 1e-12, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--numbers", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    E = os.path.abspath(a.exp)
    old = json.load(open(a.numbers))
    new = copy.deepcopy(old)
    missing, found = [], 0

    def setv(path, value, src):
        nonlocal found
        d = new
        for p in path[:-1]:
            d = d[p]
        if value is None or (isinstance(value, list) and any(v is None for v in value)):
            missing.append("%s  <- %s" % ("/".join(map(str, path)), src))
            return
        d[path[-1]] = value
        found += 1

    def test(tag, key, k, final=False, metric="test_mean_dice"):
        r = load(E, tag, key, k)
        if r is None:
            return None
        if final:
            return r2((r.get("final_weights") or {}).get(metric, r.get(metric)))
        return r2(r.get(metric))

    def three(tag, key, final=False, metric="test_mean_dice"):
        return [test(tag, key, k, final, metric) for k in range(3)]

    def val(d):
        f = os.path.join(d, "results.json")
        return r2(json.load(open(f)).get("val_dice")) if os.path.exists(f) else None

    M = "monuseg_50"
    for tag, path in (("main_sup", ("monuseg", "supervised_50_s0")),):
        r = load(E, tag, M, 0)
        for sub, kk in (("nuclei_f1", "test_mean_dice"), ("fgbg_f1", "selfloop_f1_fg_bg_mean"),
                        ("bal_acc", "balanced_accuracy")):
            setv(path + (sub,), r2(r.get(kk)) if r else None, "%s/%s_s0" % (tag, M))
    sup100 = None
    for f in sorted(glob.glob(os.path.join(E, "*", "monuseg_100_s0", "results.json"))):
        c = yaml.safe_load(open(os.path.join(os.path.dirname(f), "config.yaml")))
        if int(c.get("warmup_iters", 0)) >= int(c.get("total_iters", 1)):
            sup100 = json.load(open(f))
            break
    for sub, kk in (("nuclei_f1", "test_mean_dice"), ("fgbg_f1", "selfloop_f1_fg_bg_mean"),
                    ("bal_acc", "balanced_accuracy")):
        setv(("monuseg", "supervised_100_s0", sub), r2(sup100.get(kk)) if sup100 else None, "*/monuseg_100_s0 (supervised)")
    setv(("monuseg", "recipe"), three("mabl_none", M), "mabl_none/%s_s{0,1,2}" % M)
    setv(("monuseg", "cssl_full"), three("cssl_strongest", M), "cssl_strongest/%s_s{0,1,2}" % M)
    r = load(E, "cssl_strongest", M, 0)
    for sub, kk in (("pooled_nuclei_f1", "pooled_dice_nucleus"), ("fgbg_f1", "selfloop_f1_fg_bg_mean"),
                    ("bal_acc", "balanced_accuracy")):
        setv(("monuseg", "cssl_full_secondary", sub), r2(r.get(kk)) if r else None, "cssl_strongest/%s_s0" % M)
    for name, tag in (("Recipe only", "mabl_none"), ("w/o CL", "sabl_no_cl"), ("w/o CR", "sabl_no_cr"),
                      ("w/o HNM", "sabl_no_hnm"), ("w/o Silhouette", "sabl_fixlam"), ("w/o memory bank", "sabl_no_mem"),
                      ("w/o boundary class", "sabl_no_bnd"), ("w/o feature-level embedding", "sabl_no_v2"),
                      ("Full", "cssl_strongest")):
        setv(("monuseg", "ablation_s0", name), test(tag, M, 0), "%s/%s_s0" % (tag, M))

    C = "chaos_25"
    for key, tag in (("supervised", "main_sup"), ("recipe", "cabl_none"), ("cssl_full", "cssl_strongest")):
        setv(("chaos", key, "best_val"), three(tag, C), "%s/%s_s{0,1,2}" % (tag, C))
        setv(("chaos", key, "best_val_round"), three(tag, C, metric="dafnet_dice_round"),
             "%s/%s_s{0,1,2} dafnet_dice_round" % (tag, C))
    for name, tag in (("Recipe only", "cabl_none"), ("w/o CL", "sabl_no_cl"), ("w/o CR", "sabl_no_cr"),
                      ("w/o HNM", "sabl_no_hnm"), ("w/o Silhouette", "sabl_fixlam"), ("Full", "cssl_strongest")):
        setv(("chaos", "ablation_s1", name), test(tag, C, 1), "%s/%s_s1" % (tag, C))

    diffs = []

    def walk(o, n, path):
        if isinstance(o, dict):
            for k in o:
                walk(o[k], n.get(k) if isinstance(n, dict) else None, path + [k])
        elif isinstance(o, list):
            for i, (x, y) in enumerate(zip(o, n or [])):
                walk(x, y, path + [str(i)])
        elif isinstance(o, (int, float)) and isinstance(n, (int, float)):
            if abs(o - n) > 0.005:
                diffs.append(("/".join(path), o, n))
        elif o is None and n is not None:
            diffs.append(("/".join(path), o, n))

    walk(old, new, [])
    print("values read from run directories: %d" % found)
    if missing:
        print("\nkept from the current numbers.json (run directory not found): %d" % len(missing))
        for m in missing:
            print("  " + m)
    if diffs:
        print("\nvalues that DIFFER from the current numbers.json:")
        for p, o, n in diffs:
            print("  %-55s %s -> %s" % (p, o, n))
    else:
        print("\nall values found agree with the current numbers.json (to 0.005)")
    if a.write:
        tmp = a.numbers + ".tmp"
        with open(tmp, "w") as f:
            json.dump(new, f, indent=2)
        os.replace(tmp, a.numbers)
        print("\nwrote %s; now run: python paper/gen_numbers.py && python paper/make_figures.py" % a.numbers)
    elif any(o is not None for _, o, _ in diffs):
        sys.exit(1)


if __name__ == "__main__":
    main()
