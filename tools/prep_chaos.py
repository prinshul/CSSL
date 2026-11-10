#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image

RANGES = [(1, 55, 70), (2, 110, 135), (3, 175, 200), (4, 240, 255)]
NAMES = ["background", "liver", "right_kidney", "left_kidney", "spleen"]


def map_labels(png: np.ndarray) -> np.ndarray:
    out = np.zeros(png.shape, np.uint8)
    for c, lo, hi in RANGES:
        out[(png >= lo) & (png <= hi)] = c
    return out


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def read_subject(t2_dir):
    import pydicom
    dcm_files = glob.glob(os.path.join(t2_dir, "DICOM_anon", "*.dcm"))
    png_files = sorted(glob.glob(os.path.join(t2_dir, "Ground", "*.png")), key=natural_key)
    if not dcm_files or len(dcm_files) != len(png_files):
        raise RuntimeError("%s: %d DICOM vs %d masks" % (t2_dir, len(dcm_files), len(png_files)))
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in png_files}
    dcms = [(pydicom.dcmread(f), f) for f in dcm_files]
    stems = [os.path.splitext(os.path.basename(f))[0] for _, f in dcms]
    if all(s in by_stem for s in stems):
        dcms.sort(key=lambda t: int(getattr(t[0], "InstanceNumber", 0)))
        pairs = [(d, by_stem[os.path.splitext(os.path.basename(f))[0]]) for d, f in dcms]
    else:
        dcms.sort(key=lambda t: natural_key(t[1]))
        pairs = list(zip([d for d, _ in dcms], png_files))
    img = np.stack([d.pixel_array.astype(np.float32) for d, _ in pairs])
    lab = np.stack([map_labels(np.array(Image.open(p).convert("L"))) for _, p in pairs])
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return img.astype(np.float16), lab


def overlay(img, lab, path):
    rgb = np.repeat((img.astype(np.float32) * 255).astype(np.uint8)[..., None], 3, -1)
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}
    for c, col in colors.items():
        m = lab == c
        rgb[m] = (0.5 * rgb[m] + 0.5 * np.array(col)).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="unzipped CHAOS_Train_Sets directory")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t2_dirs = sorted(glob.glob(os.path.join(a.src, "**", "MR", "*", "T2SPIR"), recursive=True), key=natural_key)
    if not t2_dirs:
        raise SystemExit("no MR/<subject>/T2SPIR folders under %s" % a.src)
    os.makedirs(os.path.join(a.out, "overlays"), exist_ok=True)
    meta, left_votes = {}, []
    for d in t2_dirs:
        sid = os.path.basename(os.path.dirname(d))
        img, lab = read_subject(d)
        np.savez_compressed(os.path.join(a.out, "%s.npz" % sid), image=img, label=lab)
        counts = [int((lab == c).sum()) for c in range(5)]
        meta[sid] = {"slices": int(img.shape[0]), "shape": list(img.shape[1:]), "voxels": counts}
        mid = int(np.argmax((lab == 1).sum((1, 2))))
        overlay(img[mid], lab[mid], os.path.join(a.out, "overlays", "%s_slice%02d.png" % (sid, mid)))
        _, _, xs = np.nonzero(lab == 1)
        if xs.size:
            left_votes.append(xs.mean() < lab.shape[2] / 2)
        print("%-4s %3d slices %s  voxels per class %s" % (sid, img.shape[0], tuple(img.shape[1:]), counts))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    missing = [s for s, m in meta.items() if min(m["voxels"][1:]) == 0]
    print("subjects: %d; subjects missing an organ: %s" % (len(meta), missing or "none"))
    share = float(np.mean(left_votes)) if left_votes else 0.0
    print("liver on the image's left in %.0f%% of subjects%s" % (100 * share,
          "" if share >= 0.8 else "  <-- check orientation in the overlays"))


if __name__ == "__main__":
    main()
