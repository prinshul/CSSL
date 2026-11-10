#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os

import numpy as np
from PIL import Image

VOLUMES = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]
GREY = {63: 1, 126: 2, 189: 3, 252: 4}


def dafnet_order(files):
    return sorted(files, key=lambda x: os.path.basename(x).split("-")[-1], reverse=True)


def resample(img2d, res, binary, new=1.89):
    import cv2
    h, w = img2d.shape
    nh, nw = max(1, int(round(h * res[0] / new))), max(1, int(round(w * res[1] / new)))
    return cv2.resize(img2d.astype(np.float32), (nw, nh),
                      interpolation=cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR)


def load_volume(subj_dir, modality):
    import pydicom
    if modality == "t1":
        folder = os.path.join(subj_dir, "T1DUAL")
        image_folder = os.path.join(folder, "DICOM_anon", "OutPhase")
    else:
        folder = os.path.join(subj_dir, "T2SPIR")
        image_folder = os.path.join(folder, "DICOM_anon")
    image_files = dafnet_order([f for f in glob.glob(os.path.join(image_folder, "*")) if os.path.isfile(f)])
    label_files = dafnet_order(glob.glob(os.path.join(folder, "Ground", "*")))
    if len(image_files) != len(label_files) or not image_files:
        raise RuntimeError("%s %s: %d images vs %d masks" % (subj_dir, modality, len(image_files), len(label_files)))
    dcm = [pydicom.dcmread(f) for f in image_files]
    res = [float(v) for v in dcm[0].PixelSpacing]
    images = np.stack([resample(d.pixel_array, res, False) for d in dcm])
    labels = np.stack([resample(np.array(Image.open(f).convert("L")), res, True) for f in label_files])
    lab = np.zeros(labels.shape, np.uint8)
    for g, c in GREY.items():
        lab[labels == g] = c
    return images, lab, res


def select_slices(v, images_t1, labels_t1, images_t2, labels_t2):
    cat = np.concatenate
    if v == 1:
        images_t2 = images_t2[1:]; labels_t2 = labels_t2[1:]
        images_t1 = images_t1[0:26]; labels_t1 = labels_t1[0:26]
        images_t2 = images_t2[4:24]; labels_t2 = labels_t2[4:24]
        images_t1 = cat([images_t1[0:5], images_t1[7:10], images_t1[13:17], images_t1[18:]], axis=0)
        labels_t1 = cat([labels_t1[0:5], labels_t1[7:10], labels_t1[13:17], labels_t1[18:]], axis=0)
    if v == 2:
        images_t1 = cat([images_t1[4:7], images_t1[8:23]], axis=0)
        labels_t1 = cat([labels_t1[4:7], labels_t1[8:23]], axis=0)
        images_t2 = images_t2[3:22]; labels_t2 = labels_t2[3:22]
        images_t1 = cat([images_t1[0:11], images_t1[12:18]], axis=0)
        labels_t1 = cat([labels_t1[0:11], labels_t1[12:18]], axis=0)
        images_t2 = cat([images_t2[0:11], images_t2[12:18]], axis=0)
        labels_t2 = cat([labels_t2[0:11], labels_t2[12:18]], axis=0)
    if v == 3:
        images_t1 = cat([images_t1[11:14], images_t1[15:26]], axis=0)
        labels_t1 = cat([labels_t1[11:14], labels_t1[15:26]], axis=0)
        images_t2 = images_t2[9:23]; labels_t2 = labels_t2[9:23]
    if v == 5:
        images_t1 = cat([images_t1[4:5], images_t1[8:24]], axis=0)
        labels_t1 = cat([labels_t1[4:5], labels_t1[8:24]], axis=0)
        images_t2 = images_t2[2:22]; labels_t2 = labels_t2[2:22]
        images_t2 = cat([images_t2[0:6], images_t2[9:]], axis=0)
        labels_t2 = cat([labels_t2[0:6], labels_t2[9:]], axis=0)
        images_t1 = cat([images_t1[0:8], images_t1[9:]], axis=0)
        labels_t1 = cat([labels_t1[0:8], labels_t1[9:]], axis=0)
        images_t2 = cat([images_t2[0:8], images_t2[9:]], axis=0)
        labels_t2 = cat([labels_t2[0:8], labels_t2[9:]], axis=0)
    if v == 8:
        images_t1 = images_t1[2:-2]; labels_t1 = labels_t1[2:-2]
        images_t1 = cat([images_t1[5:11], images_t1[12:27]], axis=0)
        labels_t1 = cat([labels_t1[5:11], labels_t1[12:27]], axis=0)
        images_t2 = images_t2[6:27]; labels_t2 = labels_t2[6:27]
    if v == 10:
        images_t1 = images_t1[14:38]; labels_t1 = labels_t1[14:38]
        images_t2 = images_t2[5:24]; labels_t2 = labels_t2[5:24]
        images_t1 = cat([images_t1[0:8], images_t1[12:18], images_t1[19:]], axis=0)
        labels_t1 = cat([labels_t1[0:8], labels_t1[12:18], labels_t1[19:]], axis=0)
    if v == 13:
        images_t1 = images_t1[4:29]; labels_t1 = labels_t1[4:29]
        images_t2 = images_t2[3:28]; labels_t2 = labels_t2[3:28]
    if v == 15:
        images_t1 = images_t1[:22]; labels_t1 = labels_t1[:22]
        images_t2 = images_t2[:22]; labels_t2 = labels_t2[:22]
    if v == 19:
        images_t1 = images_t1[8:27]; labels_t1 = labels_t1[8:27]
        images_t2 = images_t2[5:24]; labels_t2 = labels_t2[5:24]
    if v == 20:
        images_t1 = images_t1[2:21]; labels_t1 = labels_t1[2:21]
        images_t2 = images_t2[2:21]; labels_t2 = labels_t2[2:21]
    if v == 21:
        images_t1 = images_t1[3:19]; labels_t1 = labels_t1[3:19]
        images_t2 = images_t2[5:21]; labels_t2 = labels_t2[5:21]
    if v == 22:
        images_t1 = images_t1[:-2]; labels_t1 = labels_t1[:-2]
        images_t1 = cat([images_t1[8:17], images_t1[18:26]], axis=0)
        labels_t1 = cat([labels_t1[8:17], labels_t1[18:26]], axis=0)
        images_t2 = cat([images_t2[3:12], images_t2[15:23]], axis=0)
        labels_t2 = cat([labels_t2[3:12], labels_t2[15:23]], axis=0)
    if v == 31:
        images_t1 = images_t1[7:23]; labels_t1 = labels_t1[7:23]
        images_t2 = cat([images_t2[5:12], images_t2[13:22]], axis=0)
        labels_t2 = cat([labels_t2[5:12], labels_t2[13:22]], axis=0)
    if v == 32:
        images_t1 = images_t1[5:32]; labels_t1 = labels_t1[5:32]
        images_t2 = images_t2[3:30]; labels_t2 = labels_t2[3:30]
    if v == 33:
        images_t1 = images_t1[7:-5]; labels_t1 = labels_t1[7:-5]
        images_t2 = cat([images_t2[3:12], images_t2[15:-2]], axis=0)
        labels_t2 = cat([labels_t2[3:12], labels_t2[15:-2]], axis=0)
    if v == 34:
        images_t1 = cat([images_t1[1:2], images_t1[3:4], images_t1[5:6], images_t1[7:27]], axis=0)
        labels_t1 = cat([labels_t1[1:2], labels_t1[3:4], labels_t1[5:6], labels_t1[7:27]], axis=0)
        images_t1 = cat([images_t1[0:14], images_t1[15:16], images_t1[17:18], images_t1[19:22], images_t1[23:24]], axis=0)
        labels_t1 = cat([labels_t1[0:14], labels_t1[15:16], labels_t1[17:18], labels_t1[19:22], labels_t1[23:24]], axis=0)
        images_t2 = images_t2[2:21]; labels_t2 = labels_t2[2:21]
    if v == 36:
        images_t1 = images_t1[8:25]; labels_t1 = labels_t1[8:25]
        images_t2 = cat([images_t2[4:6], images_t2[7:22]], axis=0)
        labels_t2 = cat([labels_t2[4:6], labels_t2[7:22]], axis=0)
    if v == 37:
        images_t1 = cat([images_t1[9:23], images_t1[24:-1]], axis=0)
        labels_t1 = cat([labels_t1[9:23], labels_t1[24:-1]], axis=0)
        images_t2 = cat([images_t2[4:6], images_t2[7:21], images_t2[22:-7]], axis=0)
        labels_t2 = cat([labels_t2[4:6], labels_t2[7:21], labels_t2[22:-7]], axis=0)
    if v == 38:
        images_t1 = images_t1[9:24]; labels_t1 = labels_t1[9:24]
        images_t2 = images_t2[9:24]; labels_t2 = labels_t2[9:24]
    if v == 39:
        images_t1 = images_t1[3:22]; labels_t1 = labels_t1[3:22]
        images_t2 = images_t2[3:22]; labels_t2 = labels_t2[3:22]
    return images_t1, labels_t1, images_t2, labels_t2


def rescale01(s):
    lo, hi = float(s.min()), float(s.max())
    return np.zeros_like(s) if hi == lo else (s - lo) / (hi - lo)


def crop_same(x, size, pad_value_edge=True):
    for dim in (1, 2):
        n = x.shape[dim]
        if n > size:
            l = int(math.ceil((n - size) / 2))
            x = x[:, l:n - l] if dim == 1 else x[:, :, l:n - l]
        n = x.shape[dim]
        if n < size:
            l = int((size - n) / 2)
            pw = [(0, 0), (0, 0), (0, 0)]
            pw[dim] = (l, size - n - l)
            x = np.pad(x, pw, mode="edge")
    return x


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
    p.add_argument("--size", type=int, default=192, help="crop size (DAFNet: 192)")
    p.add_argument("--allow-mismatch", action="store_true", help="tests on synthetic data only")
    a = p.parse_args()
    mr = sorted(glob.glob(os.path.join(a.src, "**", "MR"), recursive=True))
    if not mr:
        raise SystemExit("no MR folder under %s" % a.src)
    mr = mr[0]
    os.makedirs(os.path.join(a.out, "overlays"), exist_ok=True)
    meta, bad = {}, []
    for v in VOLUMES:
        sd = os.path.join(mr, str(v))
        i1, l1, r1 = load_volume(sd, "t1")
        i2, l2, r2 = load_volume(sd, "t2")
        n1, n2 = len(i1), len(i2)
        i1, l1, i2, l2 = select_slices(v, i1, l1, i2, l2)
        if len(i1) != len(i2):
            bad.append(v)
        for kind, img, lab, res, n in (("t1", i1, l1, r1, n1), ("t2", i2, l2, r2, n2)):
            img = np.stack([rescale01(s) for s in img])
            img, lab = crop_same(img, a.size), crop_same(lab, a.size)
            np.savez_compressed(os.path.join(a.out, "%d_%s.npz" % (v, kind)), image=img.astype(np.float16), label=lab)
            counts = [int((lab == c).sum()) for c in range(5)]
            meta["%d_%s" % (v, kind)] = {"slices_read": n, "slices_kept": int(img.shape[0]), "spacing_mm": res,
                                         "voxels": counts}
            mid = int(np.argmax((lab == 1).sum((1, 2))))
            overlay(img[mid], lab[mid], os.path.join(a.out, "overlays", "%d_%s_slice%02d.png" % (v, kind, mid)))
        print("%-3d T1 %2d -> %2d slices (%.2f mm), T2 %2d -> %2d slices (%.2f mm)%s" % (
            v, n1, len(i1), r1[0], n2, len(i2), r2[0], "" if len(i1) == len(i2) else "   <-- T1/T2 counts differ"))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    left = [np.nonzero(np.load(os.path.join(a.out, "%d_t2.npz" % v))["label"] == 1)[2].mean() < a.size / 2
            for v in VOLUMES]
    print("T2: liver on the image's left in %.0f%% of subjects" % (100 * np.mean(left)))
    if bad:
        print("ERROR: T1 and T2 slice counts differ for subjects %s. DAFNet's slice ranges pair T1 and T2 slices "
              "one to one (its code cannot run otherwise), so the files were not read in DAFNet's order or the "
              "download differs. Send this log." % bad)
        if not a.allow_mismatch:
            os.replace(os.path.join(a.out, "meta.json"), os.path.join(a.out, "meta_MISMATCH.json"))
            raise SystemExit(2)
    else:
        print("T1 and T2 slice counts match for all 20 subjects (DAFNet's pairing reproduced).")


if __name__ == "__main__":
    main()
