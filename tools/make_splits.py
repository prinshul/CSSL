#!/usr/bin/env python3

import argparse
import json
import os
import random

CHAOS_LABELED = {"50": 7, "25": 3, "13": 2}
DAFNET_SPLITS = [
    {"validation": [31, 36, 13], "test": [10, 22, 34], "training": [5, 3, 1, 15, 19, 2, 20, 37, 32, 38, 8, 39, 21, 33]},
    {"validation": [13, 3, 20], "test": [5, 15, 39], "training": [33, 8, 38, 34, 36, 31, 32, 37, 22, 2, 1, 10, 19, 21]},
    {"validation": [37, 13, 33], "test": [1, 19, 32], "training": [5, 20, 31, 2, 38, 3, 8, 15, 22, 10, 34, 39, 36, 21]},
]
MONUSEG_LABELED = {"20": 6, "50": 15}
MONUSEG_TRAIN_30 = {
    "TCGA-A7-A13E", "TCGA-A7-A13F", "TCGA-AR-A1AK", "TCGA-AR-A1AS", "TCGA-E2-A14V", "TCGA-E2-A1B5",
    "TCGA-B0-5698", "TCGA-B0-5710", "TCGA-B0-5711", "TCGA-HE-7128", "TCGA-HE-7129", "TCGA-HE-7130",
    "TCGA-18-5592", "TCGA-21-5784", "TCGA-21-5786", "TCGA-38-6178", "TCGA-49-4488", "TCGA-50-5931",
    "TCGA-CH-5767", "TCGA-G9-6336", "TCGA-G9-6348", "TCGA-G9-6356", "TCGA-G9-6362", "TCGA-G9-6363",
    "TCGA-DK-A2I6", "TCGA-G2-A2EK", "TCGA-AY-A8YK", "TCGA-NH-A8F7", "TCGA-KB-A93J", "TCGA-RD-A8N9",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["chaos", "monuseg"], required=True)
    p.add_argument("--root", required=True, help="output of prep_chaos.py / prep_monuseg.py")
    p.add_argument("--protocol", choices=["seeds", "dafnet"], default="seeds",
                   help="CHAOS: seeds = three random 14/3/3 partitions; dafnet = DAFNet's splits and labeled subjects")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits"))
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, a.dataset), exist_ok=True)
    if a.dataset == "chaos":
        names = [f[:-4] for f in os.listdir(a.root) if f.endswith(".npz")]
        ids = sorted({n[:-3] if n.endswith(("_t1", "_t2")) else n for n in names}, key=lambda s: (len(s), s))
        assert len(ids) == 20, "expected 20 CHAOS MR subjects, found %d" % len(ids)
        if a.protocol == "dafnet":
            import numpy as np
            for k, sp in enumerate(DAFNET_SPLITS):
                vols = sorted(sp["training"])
                labeled = {}
                for f, l_mix in (("50", 0.5), ("25", 0.25), ("13", 0.125)):
                    np.random.seed(10)
                    pick = np.random.choice(vols, size=int(np.round(l_mix * len(vols))), replace=False)
                    labeled[f] = [str(v) for v in pick.tolist()]
                split = {"test": [str(v) for v in sp["test"]], "val": [str(v) for v in sp["validation"]],
                         "train": [str(v) for v in sp["training"]], "labeled": labeled}
                assert set(split["train"]) <= set(ids), "subjects missing in %s" % a.root
                json.dump(split, open(os.path.join(a.out, a.dataset, "dafnet%d.json" % k), "w"), indent=1)
                print("dafnet split %d: test %s, val %s, labeled at 25%%: %s" % (k, split["test"], split["val"],
                                                                              labeled["25"]))
            return
        for seed in range(3):
            r = random.Random(seed)
            order = ids[:]
            r.shuffle(order)
            test, val, train = order[:3], order[3:6], order[6:]
            split = {"test": test, "val": val, "train": train,
                     "labeled": {k: train[:n] for k, n in CHAOS_LABELED.items()}}
            json.dump(split, open(os.path.join(a.out, a.dataset, "seed%d.json" % seed), "w"), indent=1)
    else:
        found = sorted(f[:-4] for f in os.listdir(os.path.join(a.root, "train", "images")) if f.endswith(".png"))
        test = sorted(f[:-4] for f in os.listdir(os.path.join(a.root, "test", "images")) if f.endswith(".png"))
        if len(found) == 30:
            train, extra = found, []
        else:
            train = [n for n in found if n[:12] in MONUSEG_TRAIN_30]
            extra = [n for n in found if n[:12] not in MONUSEG_TRAIN_30]
        if extra:
            print("MoNuSeg: using the 30 original training images; not used (later additions): %s" % ", ".join(extra))
        assert len(train) == 30 and len(test) == 14, (
            "expected the 30 original training and 14 test MoNuSeg images, found %d (of %d) / %d"
            % (len(train), len(found), len(test)))
        for seed in range(3):
            order = train[:]
            random.Random(seed).shuffle(order)
            split = {"test": test, "train": train, "labeled": {k: order[:n] for k, n in MONUSEG_LABELED.items()}}
            json.dump(split, open(os.path.join(a.out, a.dataset, "seed%d.json" % seed), "w"), indent=1)
    print("wrote", os.path.join(a.out, a.dataset))


if __name__ == "__main__":
    main()
