import math
import random

import numpy as np

try:
    import cv2
    cv2.setNumThreads(1)
except ImportError:
    cv2 = None

def _affine_matrix(h, w, out, rot_deg, scale, shift):
    sx, sy = out / w, out / h
    R = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]], np.float64)
    c = (out - 1) / 2.0
    t = math.radians(rot_deg)
    A = scale * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    T = np.eye(3)
    T[:2, :2] = A
    T[:2, 2] = np.array([c, c]) - A @ np.array([c, c]) + np.asarray(shift, np.float64) * out
    return (T @ R)[:2]


def affine_resize(img, mask, out, rot_deg=0.0, scale=1.0, shift=(0.0, 0.0)):
    C, h, w = img.shape
    M = _affine_matrix(h, w, out, rot_deg, scale, shift)
    if cv2 is not None:
        chans = [cv2.warpAffine(np.ascontiguousarray(img[c]), M, (out, out), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in range(C)]
        img_o = np.stack(chans).astype(np.float32)
        mask_o = None if mask is None else cv2.warpAffine(
            np.ascontiguousarray(mask.astype(np.uint8)), M, (out, out), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_o, mask_o
    from scipy import ndimage
    M3 = np.vstack([M, [0, 0, 1]])
    inv = np.linalg.inv(M3)
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
    inv_rc = P @ inv @ P
    mat, off = inv_rc[:2, :2], inv_rc[:2, 2]
    img_o = np.stack([ndimage.affine_transform(img[c], mat, off, (out, out), order=1, cval=0.0)
                      for c in range(C)]).astype(np.float32)
    mask_o = None if mask is None else ndimage.affine_transform(mask.astype(np.uint8), mat, off, (out, out),
                                                                order=0, cval=0)
    return img_o, mask_o


def random_affine_params(rot=15.0, scale=(0.85, 1.2), shift=0.08, p=0.8):
    if random.random() >= p:
        return 0.0, 1.0, (0.0, 0.0)
    s = math.exp(random.uniform(math.log(scale[0]), math.log(scale[1])))
    return random.uniform(-rot, rot), s, (random.uniform(-shift, shift), random.uniform(-shift, shift))


def _blur(img, sigma):
    if cv2 is not None:
        return np.stack([cv2.GaussianBlur(c, (0, 0), sigma) for c in img])
    from scipy import ndimage
    return np.stack([ndimage.gaussian_filter(c, sigma) for c in img])


def mr_intensity(img, strength):
    img = img.astype(np.float32, copy=True)
    if strength == "light":
        if random.random() < 0.5:
            m = float(img.mean())
            img = (img - m) * random.uniform(0.85, 1.15) + m + random.uniform(-0.05, 0.05)
        if random.random() < 0.3:
            img = np.clip(img, 0, 1) ** random.uniform(0.8, 1.25)
    elif strength == "strong":
        if random.random() < 0.8:
            m = float(img.mean())
            img = (img - m) * random.uniform(0.6, 1.4) + m + random.uniform(-0.2, 0.2)
        if random.random() < 0.5:
            img = np.clip(img, 0, 1) ** random.uniform(0.6, 1.6)
        if random.random() < 0.5:
            img = _blur(img, random.uniform(0.1, 1.5))
        if random.random() < 0.3:
            rng = np.random.default_rng(random.getrandbits(32))
            img = img + rng.standard_normal(img.shape, dtype=np.float32) * random.uniform(0.0, 0.05)
    else:
        raise ValueError(strength)
    return np.clip(img, 0, 1).astype(np.float32)


RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], np.float64)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
HED_LIGHT = (0.1, 0.02)
HED_STRONG = (0.3, 0.05)


def hed_jitter(img_uint8, a, b):
    rgb = np.maximum(img_uint8.astype(np.float32) / 255.0, 1.0 / 255.0)
    od = -np.log(rgb).reshape(-1, 3) @ HED_FROM_RGB.astype(np.float32)
    alpha = np.random.uniform(1 - a, 1 + a, 3).astype(np.float32)
    beta = np.random.uniform(-b, b, 3).astype(np.float32)
    od = (od * alpha + beta) @ RGB_FROM_HED.astype(np.float32)
    out = np.exp(-od).reshape(img_uint8.shape) * 255.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def scaled_crop(img, mask, size, scale=(1.0, 1.0)):
    h, w = mask.shape
    s = math.exp(random.uniform(math.log(scale[0]), math.log(scale[1]))) if scale[0] != scale[1] else scale[0]
    cs = int(round(size / s))
    cs = max(8, min(cs, h, w))
    y, x = random.randint(0, h - cs), random.randint(0, w - cs)
    img, mask = img[y:y + cs, x:x + cs], mask[y:y + cs, x:x + cs]
    if cs != size:
        if cv2 is not None:
            img = cv2.resize(np.ascontiguousarray(img), (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(np.ascontiguousarray(mask), (size, size), interpolation=cv2.INTER_NEAREST)
        else:
            from PIL import Image
            img = np.array(Image.fromarray(np.ascontiguousarray(img)).resize((size, size), Image.BILINEAR))
            mask = np.array(Image.fromarray(np.ascontiguousarray(mask)).resize((size, size), Image.NEAREST))
    return img, mask
