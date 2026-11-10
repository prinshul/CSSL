#!/usr/bin/env python3

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image


def xml_to_mask(xml_path, shape):
    mask = np.zeros(shape[:2], np.uint8)
    root = ET.parse(xml_path).getroot()
    n = 0
    for region in root.iter("Region"):
        pts = [(float(v.get("X")), float(v.get("Y"))) for v in region.iter("Vertex")]
        if len(pts) < 3:
            continue
        cv2.fillPoly(mask, [np.round(np.array(pts)).astype(np.int32)], 1)
        n += 1
    return mask, n


def read_rgb(path):
    try:
        import tifffile
        img = tifffile.imread(path)
    except Exception:
        img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    return img[..., :3].astype(np.uint8)


def convert(pairs, out_dir, tag, ov_dir):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    for i, (tif, xml) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(tif))[0]
        img = read_rgb(tif)
        mask, n = xml_to_mask(xml, img.shape)
        Image.fromarray(img).save(os.path.join(out_dir, "images", name + ".png"))
        Image.fromarray(mask).save(os.path.join(out_dir, "masks", name + ".png"))
        if i < 2:
            ov = img.copy()
            ov[mask == 1] = (0.5 * ov[mask == 1] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            Image.fromarray(ov).save(os.path.join(ov_dir, "%s_%s.png" % (tag, name)))
        print("%s %-28s %s  %4d nuclei  foreground %.1f%%" % (tag, name, img.shape, n, 100 * mask.mean()))


def find_pairs(tifs):
    pairs = []
    for t in sorted(tifs):
        stem = os.path.splitext(os.path.basename(t))[0]
        cands = glob.glob(os.path.join(os.path.dirname(os.path.dirname(t)), "**", stem + ".xml"), recursive=True) + \
            glob.glob(os.path.join(os.path.dirname(t), stem + ".xml"))
        if not cands:
            raise SystemExit("no XML annotation for %s" % t)
        pairs.append((t, cands[0]))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tifs = glob.glob(os.path.join(a.src, "**", "*.tif"), recursive=True)
    train = [t for t in tifs if "Tissue Images" in t]
    test = [t for t in tifs if "Tissue Images" not in t and "test" in t.lower()]
    print("found %d training and %d test images (expect 30 and 14)" % (len(train), len(test)))
    if not train or not test:
        raise SystemExit("could not find both the training and the test images under %s" % a.src)
    ov = os.path.join(a.out, "overlays")
    os.makedirs(ov, exist_ok=True)
    convert(find_pairs(train), os.path.join(a.out, "train"), "train", ov)
    convert(find_pairs(test), os.path.join(a.out, "test"), "test", ov)


if __name__ == "__main__":
    main()
