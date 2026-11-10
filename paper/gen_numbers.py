#!/usr/bin/env python3
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "numbers.json")))
m, c = N["monuseg"], N["chaos"]
out = {}


def put(name, v, nd=2, sign=False):
    if v is None:
        out[name] = "\\textit{n/a}"
        return
    s = ("%+." + str(nd) + "f") % v if sign else ("%." + str(nd) + "f") % v
    out[name] = s.replace("-", "\\textminus{}") if sign or v < 0 else s
    if sign:
        out[name + "Abs"] = ("%." + str(nd) + "f") % abs(v)


def mean(v):
    return float(np.mean(v))


def wins(a, b):
    return int(sum(x > y for x, y in zip(a, b)))


put("SL", m["selfloop_published"]); put("SLsupFifty", m["selfloop_published_sup50"])
put("SLsupHundred", m["selfloop_published_sup100"])
put("MsupFifty", m["supervised_50_s0"]["nuclei_f1"]); put("MsupHundred", m["supervised_100_s0"]["nuclei_f1"])
put("MsupFiftyFgbg", m["supervised_50_s0"]["fgbg_f1"]); put("MsupFiftyBacc", m["supervised_50_s0"]["bal_acc"])
put("MsupHundredFgbg", m["supervised_100_s0"]["fgbg_f1"]); put("MsupHundredBacc", m["supervised_100_s0"]["bal_acc"])
for tag, v in (("Mrec", m["recipe"]), ("Mcssl", m["cssl_full"])):
    for k, x in enumerate(v):
        put(tag + "s" + "abc"[k], x)
    put(tag + "Mean", mean(v)); put(tag + "Sd", float(np.std(v)))
d = [a - b for a, b in zip(m["cssl_full"], m["recipe"])]
put("MgainRec", mean(d), sign=True); out["MwinsRec"] = str(wins(m["cssl_full"], m["recipe"]))
for k, x in enumerate(d):
    put("MgainRec" + "abc"[k], x, sign=True)
put("MgainSL", mean(m["cssl_full"]) - m["selfloop_published"], sign=True)
out["MaboveSL"] = str(sum(x > m["selfloop_published"] for x in m["cssl_full"]))
put("MrecVsSL", mean(m["recipe"]) - m["selfloop_published"], sign=True)
put("MsupGapSL", m["supervised_50_s0"]["nuclei_f1"] - m["selfloop_published_sup50"], sign=True)
put("MsupHGapSL", m["supervised_100_s0"]["nuclei_f1"] - m["selfloop_published_sup100"], sign=True)
put("MgainSupHundred", mean(m["cssl_full"]) - m["supervised_100_s0"]["nuclei_f1"], sign=True)
A = m["ablation_s0"]
for k, nm in (("Rec", "Recipe only"), ("NoCL", "w/o CL"), ("NoCR", "w/o CR"), ("NoHNM", "w/o HNM"),
              ("NoSil", "w/o Silhouette"), ("NoMem", "w/o memory bank"), ("NoBnd", "w/o boundary class"),
              ("NoFeat", "w/o feature-level embedding"), ("Full", "Full")):
    put("MA" + k, A[nm])
    if k != "Full":
        put("MD" + k, A["Full"] - A[nm], sign=True)
sec = m.get("cssl_full_secondary") or {}
put("McsslFgbg", sec.get("fgbg_f1")); put("McsslBacc", sec.get("bal_acc")); put("McsslPooled", sec.get("pooled_nuclei_f1"))
for nm, v in m["selfloop_table2_50"].items():
    put("SLt" + "".join(ch for ch in nm.title() if ch.isalpha()), v)

put("DAF", c["dafnet_published"], nd=1)
for tag, key in (("Csup", "supervised"), ("Crec", "recipe"), ("Ccssl", "cssl_full")):
    for rule, rk in (("B", "best_val"),):
        v = c[key][rk]
        for k, x in enumerate(v):
            put(tag + rule + "abc"[k], x)
        put(tag + rule + "Mean", mean(v)); put(tag + rule + "Sd", float(np.std(v)))
for rule, rk in (("B", "best_val"),):
    a, b = c["cssl_full"][rk], c["recipe"][rk]
    put("CgainRec" + rule, mean(a) - mean(b), sign=True); out["CwinsRec" + rule] = str(wins(a, b))
    for k in range(3):
        put("CgainRec" + rule + "abc"[k], a[k] - b[k], sign=True)
    put("CgainSup" + rule, mean(c["cssl_full"][rk]) - mean(c["supervised"][rk]), sign=True)
    put("CrecVsDAF" + rule, mean(c["recipe"][rk]) - c["dafnet_published"], sign=True)
put("CgainDAF", mean(c["cssl_full"]["best_val"]) - c["dafnet_published"], sign=True)
have_round = all(c[key].get("best_val_round") for key in ("supervised", "recipe", "cssl_full"))
if have_round:
    for tag, key in (("Csup", "supervised"), ("Crec", "recipe"), ("Ccssl", "cssl_full")):
        v = c[key]["best_val_round"]
        for k, x in enumerate(v):
            put(tag + "R" + "abc"[k], x)
        put(tag + "RMean", mean(v)); put(tag + "RSd", float(np.std(v)))
        put(tag + "RDiff", mean(v) - mean(c[key]["best_val"]), sign=True)
    put("CgainRecR", mean(c["cssl_full"]["best_val_round"]) - mean(c["recipe"]["best_val_round"]), sign=True)
    put("CgainDAFR", mean(c["cssl_full"]["best_val_round"]) - c["dafnet_published"], sign=True)
    put("CroundMaxDiff", max(abs(mean(c[key]["best_val_round"]) - mean(c[key]["best_val"]))
                             for key in ("supervised", "recipe", "cssl_full")))
A = c["ablation_s1"]
for k, nm in (("Rec", "Recipe only"), ("NoCL", "w/o CL"), ("NoCR", "w/o CR"), ("NoHNM", "w/o HNM"),
              ("NoSil", "w/o Silhouette"), ("Full", "Full")):
    put("CA" + k, A[nm])
    if k != "Full":
        put("CD" + k, A["Full"] - A[nm], sign=True)
for nm, v in c["dafnet_table_25"].items():
    out["Dt" + "".join(ch for ch in nm.title() if ch.isalpha())] = ("%.1f" % v) if isinstance(v, float) else str(v)

with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    have = all(sec.get(k) is not None for k in ("fgbg_f1", "bal_acc", "pooled_nuclei_f1"))
    f.write("\\newif\\ifhavesecondary\n\\havesecondary%s\n" % ("true" if have else "false"))
    f.write("\\newif\\ifhaveround\n\\haveround%s\n" % ("true" if have_round else "false"))
    for k in sorted(out):
        assert k.isalpha(), k
        f.write("\\newcommand{\\n%s}{%s}\n" % (k, out[k]))
print("numbers.tex: %d macros" % len(out))
