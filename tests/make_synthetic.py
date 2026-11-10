#!/usr/bin/env python3

import argparse
import os

import numpy as np
from PIL import Image


def _write_dicom(path, img, inst, z, spacing, echo):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.InstanceNumber = "MR", inst
    ds.SliceLocation, ds.PixelSpacing, ds.EchoTime = z, [spacing, spacing], echo
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = img.astype(np.uint16).tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:
        ds.save_as(path, write_like_original=False)


def _chaos_label(k, size):
    lab = np.zeros((size, size), np.uint8)
    s = size / 64.0
    lab[int(8 * s):int(30 * s), int(4 * s):int((12 + 2 * k) * s)] = 63
    if k % 3 != 0:
        lab[int(34 * s):int(44 * s), int(8 * s):int(18 * s)] = 126
        lab[int(34 * s):int(44 * s), int(44 * s):int(54 * s)] = 189
    lab[int(10 * s):int((14 + k) * s), int(40 * s):int(56 * s)] = 252
    return lab


def chaos(root, n_subjects=20, slices=8, size=64, seed=0, spacing=1.6, t1_slices=None):
    rng = np.random.default_rng(seed)
    ids = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39][:n_subjects]
    for sid in ids:
        mr = os.path.join(root, "CHAOS_Train_Sets", "Train_Sets", "MR", str(sid))
        t2 = os.path.join(mr, "T2SPIR")
        os.makedirs(os.path.join(t2, "DICOM_anon"), exist_ok=True)
        os.makedirs(os.path.join(t2, "Ground"), exist_ok=True)
        t1 = os.path.join(mr, "T1DUAL")
        for d in ("DICOM_anon/InPhase", "DICOM_anon/OutPhase", "Ground"):
            os.makedirs(os.path.join(t1, d), exist_ok=True)
        for k in range(slices):
            lab = _chaos_label(k % 12, size)
            z = 100.0 - 7.0 * k
            img = (lab.astype(np.float32) * 4 + rng.normal(200, 30, lab.shape)).clip(0, 4000)
            name = "IMG-%04d-%05d" % (sid, k + 1)
            _write_dicom(os.path.join(t2, "DICOM_anon", name + ".dcm"), img, k + 1, z, spacing, 80.0)
            Image.fromarray(lab).save(os.path.join(t2, "Ground", name + ".png"))
        for k in range(t1_slices or slices):
            lab = _chaos_label(k % 12, size)
            z = 100.0 - 7.0 * k
            out_name, in_name = "IMG-%04d-%05d" % (sid, 2 * k + 1), "IMG-%04d-%05d" % (sid, 2 * k + 2)
            img1 = (1000.0 - lab.astype(np.float32) * 3 + rng.normal(0, 30, lab.shape)).clip(0, 4000)
            _write_dicom(os.path.join(t1, "DICOM_anon", "OutPhase", out_name + ".dcm"), img1 * 0.8, 2 * k + 1, z,
                         spacing, 2.3)
            _write_dicom(os.path.join(t1, "DICOM_anon", "InPhase", in_name + ".dcm"), img1, 2 * k + 2, z, spacing, 4.6)
            Image.fromarray(lab).save(os.path.join(t1, "Ground", (in_name if sid % 2 == 0 else out_name) + ".png"))


def monuseg(root, size=128, seed=0):
    import tifffile
    rng = np.random.default_rng(seed)
    for split, n, sub_img, sub_xml in (("train", 30, "MoNuSeg 2018 Training Data/Tissue Images",
                                        "MoNuSeg 2018 Training Data/Annotations"),
                                       ("test", 14, "MoNuSegTestData", "MoNuSegTestData")):
        os.makedirs(os.path.join(root, sub_img), exist_ok=True)
        os.makedirs(os.path.join(root, sub_xml), exist_ok=True)
        for i in range(n):
            name = "TCGA-SYN-%s-%02d" % (split, i)
            img = np.full((size, size, 3), (230, 200, 220), np.float32) + rng.normal(0, 8, (size, size, 3))
            regions = []
            for r in range(12):
                cx, cy, rad = rng.integers(8, size - 8), rng.integers(8, size - 8), rng.integers(3, 7)
                ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                pts = [(cx + rad * np.cos(t), cy + rad * np.sin(t)) for t in ang]
                yy, xx = np.ogrid[:size, :size]
                img[(xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2] = (90, 40, 120)
                verts = "".join('<Vertex X="%.2f" Y="%.2f" Z="0"/>' % p for p in pts)
                regions.append('<Region Id="%d"><Vertices>%s</Vertices></Region>' % (r + 1, verts))
            tifffile.imwrite(os.path.join(root, sub_img, name + ".tif"), np.clip(img, 0, 255).astype(np.uint8))
            xml = '<Annotations><Annotation Id="1"><Regions>%s</Regions></Annotation></Annotations>' % "".join(regions)
            open(os.path.join(root, sub_xml, name + ".xml"), "w").write(xml)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--which", choices=["chaos", "monuseg", "both"], default="both")
    p.add_argument("--chaos-slices", type=int, default=8, help="T2 slices per subject (T1: --chaos-t1-slices)")
    p.add_argument("--chaos-t1-slices", type=int, default=None)
    p.add_argument("--chaos-size", type=int, default=64)
    a = p.parse_args()
    if a.which in ("chaos", "both"):
        chaos(os.path.join(a.out, "chaos_raw"), slices=a.chaos_slices, size=a.chaos_size, t1_slices=a.chaos_t1_slices)
    if a.which in ("monuseg", "both"):
        monuseg(os.path.join(a.out, "monuseg_raw"))
