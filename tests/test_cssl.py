import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cssl import AdaptivePerturbation, cssl_contrastive_loss, sample_class_patches, silhouette_l1
from cssl.supcon import gaussian_aux_negatives


def test_patches_prominent_class_masking_and_min_pixels():
    img = torch.arange(128 * 128, dtype=torch.float).view(1, 1, 128, 128) + 1
    lab = torch.zeros(1, 128, 128, dtype=torch.long)
    lab[0, 0:64, 0:40] = 2
    lab[0, 0:64, 40:64] = 1
    lab[0, 100:103, 100:103] = 3
    lab[0, 120:128, :] = 255
    d = sample_class_patches(img, lab, classes=[1, 2, 3], patch=64, stride=32, min_pixels=25)
    labels = d["labels"].tolist()
    assert 3 not in labels and 2 in labels and 1 in labels
    i = labels.index(2)
    m = d["masks"][i, 0]
    assert int(m.sum()) == 64 * 40
    assert torch.equal(d["patches"][i, 0][m == 0], torch.zeros(int((m == 0).sum())))
    assert torch.equal(d["patches"][i, 0][m == 1], img[0, 0, 0:64, 0:64][m == 1])
    d_bg = sample_class_patches(img, lab, classes=[0, 1, 2], patch=64, stride=32, min_pixels=25)
    assert 0 in d_bg["labels"].tolist()


def test_patches_each_mode_gives_small_objects_their_own_patches():
    img = torch.rand(1, 3, 64, 64)
    lab = torch.zeros(1, 64, 64, dtype=torch.long)
    lab[0, 10:20, 10:20] = 1
    prom = sample_class_patches(img, lab, classes=[0, 1], patch=64, stride=32, min_pixels=25, mode="prominent")
    each = sample_class_patches(img, lab, classes=[0, 1], patch=64, stride=32, min_pixels=25, mode="each")
    assert prom["labels"].tolist() == [0]
    assert sorted(each["labels"].tolist()) == [0, 1]
    i = each["labels"].tolist().index(1)
    assert int(each["masks"][i].sum()) == 100


def test_patches_cap_is_class_balanced_and_small_images_padded():
    img = torch.rand(2, 3, 256, 256)
    lab = torch.ones(2, 256, 256, dtype=torch.long)
    lab[:, :, :48] = 2
    d = sample_class_patches(img, lab, classes=[1, 2], patch=64, stride=16, min_pixels=25, max_patches=20,
                             generator=torch.Generator().manual_seed(0))
    labs = d["labels"].tolist()
    assert len(labs) == 20 and labs.count(2) == 10
    small = sample_class_patches(torch.rand(1, 1, 40, 50), torch.ones(1, 40, 50, dtype=torch.long), [1], patch=64)
    assert small["patches"].shape == (1, 1, 64, 64)


def _eq1_bruteforce(z, labels, tau):
    total, cnt = 0.0, 0
    for i in range(len(z)):
        pos = [p for p in range(len(z)) if p != i and labels[p] == labels[i]]
        neg = [k for k in range(len(z)) if labels[k] != labels[i]]
        if not pos or not neg:
            continue
        negsum = sum(math.exp(float(z[i] @ z[k]) / tau) for k in neg)
        li = 0.0
        for p in pos:
            e = math.exp(float(z[i] @ z[p]) / tau)
            li += -math.log(e / (e + negsum))
        total += li / len(pos)
        cnt += 1
    return total / cnt


def test_contrastive_loss_matches_eq1_without_hard_negatives():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(12, 8), dim=1).requires_grad_()
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 0])
    loss, info = cssl_contrastive_loss(z, labels, tau=0.1, hard_ratio=0.0)
    assert abs(float(loss.detach()) - _eq1_bruteforce(z.detach(), labels.tolist(), 0.1)) < 1e-4
    assert info["cl_anchors"] == 11
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_hard_negatives_make_the_loss_harder_and_alpha_controls_hardness():
    torch.manual_seed(1)
    z = F.normalize(torch.randn(16, 8) + torch.tensor([3.0] + [0] * 7) * torch.tensor([1, -1] * 8)[:, None], dim=1)
    labels = torch.tensor([0, 1] * 8)
    base, _ = cssl_contrastive_loss(z, labels, hard_ratio=0.0)
    g = torch.Generator().manual_seed(0)
    mild, _ = cssl_contrastive_loss(z, labels, hard_ratio=1.0, alpha_max=0.01, generator=g)
    hard, _ = cssl_contrastive_loss(z, labels, hard_ratio=1.0, alpha_max=0.4, generator=torch.Generator().manual_seed(0))
    assert float(base) < float(mild) < float(hard)


def test_single_class_uses_gaussian_auxiliary_negatives():
    z = F.normalize(torch.randn(10, 16), dim=1).requires_grad_()
    labels = torch.zeros(10, dtype=torch.long)
    none, info0 = cssl_contrastive_loss(z, labels)
    assert info0["cl_anchors"] == 0 and float(none) == 0.0
    aux = gaussian_aux_negatives(16, 32, seed=0)
    assert torch.allclose(aux.norm(dim=1), torch.ones(32))
    assert torch.equal(aux, gaussian_aux_negatives(16, 32, seed=0))
    loss, info = cssl_contrastive_loss(z, labels, aux_negatives=aux, generator=torch.Generator().manual_seed(0))
    assert info["cl_anchors"] == 10 and float(loss) > 0
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_silhouette_matches_sklearn_manhattan():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, (30, 5)), rng.normal(3, 1, (25, 5)), rng.normal(-3, 1, (1, 5))])
    y = np.array([0] * 30 + [1] * 25 + [2])
    ref = sk.silhouette_score(x, y, metric="manhattan")
    got = silhouette_l1(torch.from_numpy(x), torch.from_numpy(y))
    assert abs(got - ref) < 1e-6
    assert silhouette_l1(torch.randn(5, 3), torch.zeros(5, dtype=torch.long)) is None


def test_adaptive_perturbation_schedule_and_scale():
    p = AdaptivePerturbation(epsilon=0.05, momentum=0.0, relative=True)
    assert p.lam == pytest.approx(0.55)
    p.update(1.0)
    assert p.lam == pytest.approx(1.05)
    p.update(None)
    assert p.lam == pytest.approx(1.05)
    torch.manual_seed(0)
    f = torch.randn(4, 8, 32, 32) * 3.0
    out = p([f])[0]
    assert abs(float((out - f).std()) - math.sqrt(1.05) * 3.0) < 0.1
    p_abs = AdaptivePerturbation(epsilon=0.0, momentum=0.0, relative=False, init_omega=-1.0)
    assert float((p_abs([f])[0] - f).abs().max()) == 0.0


def test_chaos_augmentation_keeps_sides(tmp_path):
    import random
    from data.datasets import ChaosSlices
    img = np.zeros((4, 64, 64), np.float16)
    lab = np.zeros((4, 64, 64), np.uint8)
    lab[:, 20:44, 4:24] = 1
    lab[:, 40:50, 44:52] = 3
    np.savez(tmp_path / "1.npz", image=img, label=lab)
    random.seed(0)
    np.random.seed(0)
    ds = ChaosSlices(str(tmp_path), ["1"], "train_l", size=64)
    for i in range(60):
        _, m = ds[i % len(ds)]
        m = m.numpy()
        assert np.nonzero(m == 1)[1].mean() < 32 < np.nonzero(m == 3)[1].mean()
    flipped = ChaosSlices(str(tmp_path), ["1"], "train_l", size=64, geom="unimatch")
    sides = [np.nonzero(flipped[i % 4][1].numpy() == 1)[1].mean() < 32 for i in range(60)]
    assert not all(sides)


def test_resunet18_interface():
    from model.cssl_net import CSSLNet
    from cssl.perturb import AdaptivePerturbation
    for chns, ncls in ((1, 5), (3, 2)):
        m = CSSLNet(chns, ncls, backbone="resunet18", pretrained=False).eval()
        x = torch.randn(2, chns, 64, 64)
        assert m(x).shape == (2, ncls, 64, 64)
        clean, pert = m(x, "seg_cr", perturb=AdaptivePerturbation())
        assert clean.shape == pert.shape == (2, ncls, 64, 64) and not torch.allclose(clean, pert)
        a, b = m(x, "seg_fp")
        assert a.shape == b.shape == (2, ncls, 64, 64)
        z = m(torch.randn(4, chns, 64, 64), "embed", masks=torch.ones(4, 1, 64, 64))
        assert z.shape == (4, 256) and torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-4)
        assert len(m.unet.encoder(x)) == 5


def test_resunet34_50_interface_and_input_norm():
    from model.cssl_net import CSSLNet
    from cssl.perturb import AdaptivePerturbation
    for bb, feat in (("resunet34", 512), ("resunet50", 2048)):
        m = CSSLNet(3, 5, backbone=bb, pretrained=False, input_norm="mr").eval()
        assert m.unet.feat_dim == feat and m.unet.encoder.mr_norm
        x = torch.rand(2, 3, 64, 64)
        assert m(x).shape == (2, 5, 64, 64)
        clean, pert = m(x, "seg_cr", perturb=AdaptivePerturbation())
        assert clean.shape == pert.shape == (2, 5, 64, 64)
        z = m(torch.rand(3, 3, 64, 64), "embed", masks=torch.ones(3, 1, 64, 64))
        assert z.shape == (3, 256)
    assert not CSSLNet(3, 2, backbone="resunet34", pretrained=False).unet.encoder.mr_norm
    assert CSSLNet(1, 5, backbone="resunet18", pretrained=False).unet.encoder.mr_norm
    with pytest.raises(ValueError):
        CSSLNet(3, 2, backbone="resunet34", pretrained=False, input_norm="zscore")


def test_affine_resize_identity_resize_and_scipy_fallback():
    import data.strong_aug as sa
    rng = np.random.default_rng(0)
    img = rng.random((3, 40, 48)).astype(np.float32)
    mask = rng.integers(0, 5, (40, 48)).astype(np.uint8)
    o, m = sa.affine_resize(img[:, :40, :40], mask[:40, :40], 40)
    assert np.abs(o - img[:, :40, :40]).max() < 1e-6 and (m == mask[:40, :40]).all()
    if sa.cv2 is not None:
        a = sa.affine_resize(img, mask, 64, 12.0, 1.1, (0.05, -0.03))
        cv, sa.cv2 = sa.cv2, None
        try:
            b = sa.affine_resize(img, mask, 64, 12.0, 1.1, (0.05, -0.03))
        finally:
            sa.cv2 = cv
        inner = (slice(8, -8), slice(8, -8))
        assert np.abs(a[0][(slice(None),) + inner] - b[0][(slice(None),) + inner]).mean() < 0.02
        assert (a[1][inner] != b[1][inner]).mean() < 0.03


def test_chaos_strong_25d_shapes_and_sides(tmp_path):
    import random
    from data.datasets import ChaosSlices, ChaosVolumes
    img = np.stack([np.full((64, 64), k / 10.0, np.float16) for k in range(5)])
    lab = np.zeros((5, 64, 64), np.uint8)
    lab[:, 20:44, 4:24] = 1
    lab[:, 40:50, 44:52] = 3
    np.savez(tmp_path / "1.npz", image=img, label=lab)
    random.seed(0)
    np.random.seed(0)
    ds = ChaosSlices(str(tmp_path), ["1"], "train_l", size=48, context=1, aug="strong")
    for i in range(60):
        x, m = ds[i % len(ds)]
        assert x.shape == (3, 48, 48) and m.shape == (48, 48) and x.dtype == torch.float32
        m = m.numpy()
        assert np.nonzero(m == 1)[1].mean() < 24 < np.nonzero(m == 3)[1].mean()
    u = ChaosSlices(str(tmp_path), ["1"], "train_u", size=48, context=1, aug="strong")
    w, s1, s2, b1, b2 = u[2]
    assert w.shape == s1.shape == s2.shape == (3, 48, 48) and b1.shape == (48, 48)
    assert float(w.min()) >= 0 and float(w.max()) <= 1
    vol, gt, _ = ChaosVolumes(str(tmp_path), ["1"], context=1)[0]
    assert vol.shape == (5, 3, 64, 64) and gt.shape == (5, 64, 64)
    assert float(vol[0, 0, 0, 0]) == float(vol[0, 1, 0, 0]) == 0.0
    assert abs(float(vol[2, 0, 0, 0]) - 0.1) < 1e-3 and abs(float(vol[2, 2, 0, 0]) - 0.3) < 1e-3
    with pytest.raises(ValueError):
        ChaosSlices(str(tmp_path), ["1"], "train_l", context=1)


def test_hed_jitter_identity_and_scaled_crop():
    import data.strong_aug as sa
    x = np.random.default_rng(0).integers(1, 256, (32, 32, 3)).astype(np.uint8)
    assert np.abs(sa.hed_jitter(x, 0.0, 0.0).astype(int) - x).max() <= 1
    y = sa.hed_jitter(x, *sa.HED_STRONG)
    assert y.shape == x.shape and y.dtype == np.uint8
    img = np.zeros((100, 120, 3), np.uint8)
    mask = np.zeros((100, 120), np.uint8)
    for _ in range(20):
        a, b = sa.scaled_crop(img, mask, 64, (0.8, 1.25))
        assert a.shape == (64, 64, 3) and b.shape == (64, 64)


def test_dafnet_metric_matches_its_definition():
    from train import dafnet_dice
    g = torch.zeros(3, 8, 8, dtype=torch.long)
    g[0, :4, :4] = 1
    g[1, :2, :2] = 4
    p = g.clone()
    p[1, :2, :2] = 0
    pooled, organs = dafnet_dice(p, g, 5)
    assert abs(pooled - 2 / 3) < 1e-6 and organs[0] == pytest.approx(1.0) and organs[3] == pytest.approx(2 / 3)
    q = g.clone()
    q[0, :4, :2] = 0
    assert dafnet_dice(q, g, 5)[0] == pytest.approx((2 * 8 / 24 + 1 + 1) / 3)


def test_dafnet_crop_same_and_splits():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import prep_chaos_dafnet as P
    import make_splits as M
    x = np.arange(195 * 190, dtype=np.float32).reshape(1, 195, 190)
    y = P.crop_same(x, 192)
    assert y.shape == (1, 192, 192)
    assert np.array_equal(y[0, :191, 1:191], x[0, 2:193, :])
    assert np.array_equal(y[0, 191], y[0, 190]) and np.array_equal(y[0, :, 0], y[0, :, 1])
    assert len(M.DAFNET_SPLITS) == 3
    for sp in M.DAFNET_SPLITS:
        assert len(sp["training"]) == 14 and len(sp["validation"]) == 3 and len(sp["test"]) == 3
        assert not set(sp["training"]) & set(sp["test"] + sp["validation"])
    np.random.seed(10)
    assert sorted(np.random.choice(sorted(M.DAFNET_SPLITS[0]["training"]), 4, replace=False).tolist()) == [5, 19, 20, 38]


def test_selfloop_jigsaw_inverse_and_pool():
    from cssl import selfloop as SL
    g = torch.Generator().manual_seed(0)
    pool = torch.from_numpy(SL.permutation_pool(100, seed=0))
    assert pool.shape == (100, 9) and len({tuple(p.tolist()) for p in pool}) == 100
    assert all(sorted(p.tolist()) == list(range(9)) for p in pool)
    d = (pool[:, None] != pool[None]).sum(-1).float()
    assert d[torch.triu(torch.ones(100, 100, dtype=torch.bool), 1)].mean() > 8.0
    for H, W in ((64, 64), (65, 70)):
        x = torch.rand(4, 3, H, W)
        idx, perm, rot = SL.random_jigsaw_params(4, pool, g, "cpu")
        y = SL.jigsaw(x, perm, rot)
        assert not torch.equal(y, x) and torch.allclose(SL.unjigsaw(y, perm, rot), x)
    x = torch.arange(36.0).reshape(1, 1, 6, 6)
    perm, rot = torch.arange(9)[None], torch.zeros(1, 9, dtype=torch.long)
    rot[0, 4] = 1
    y = SL.jigsaw(x, perm, rot)
    assert torch.equal(y[0, 0, 2:4, 2:4], torch.rot90(x[0, 0, 2:4, 2:4], 1, (0, 1)))
    assert torch.equal(y[0, 0, :2], x[0, 0, :2])


def test_selfloop_alg1_updates_encoder_only_and_weights():
    from cssl import selfloop as SL
    from model.cssl_net import CSSLNet
    from contextlib import nullcontext
    torch.manual_seed(0)
    m = CSSLNet(3, 2, backbone="resunet18", pretrained=False, jigsaw_classes=100).train()
    pool = torch.from_numpy(SL.permutation_pool(100, seed=0))
    enc = list(m.unet.encoder.parameters()) + list(m.jigsaw.parameters())
    opt = torch.optim.SGD(enc, lr=0.1)
    dec_before = {k: v.clone() for k, v in m.unet.decoder.state_dict().items()}
    enc_before = [p.clone() for p in m.unet.encoder.parameters()]
    x = torch.rand(2, 3, 64, 64)
    y_sl, l = SL.self_loop_pseudo_label(m, x, pool, 3, opt, torch.Generator().manual_seed(1), nullcontext)
    assert y_sl.shape == (2, 2, 64, 64) and l > 0
    assert torch.allclose(y_sl.sum(1), torch.ones(2, 64, 64), atol=1e-5)
    for k, v in m.unet.decoder.state_dict().items():
        if not k.endswith("num_batches_tracked"):
            assert torch.equal(v, dec_before[k]), k
    assert any(not torch.equal(a, b) for a, b in zip(enc_before, m.unet.encoder.parameters()))
    p = torch.zeros(1, 2, 2, 2)
    p[:, 0] = 1
    ysl = torch.zeros(1, 2, 2, 2)
    ysl[:, 0] = 1
    ysl[0, :, 0, 0] = torch.tensor([0.0, 1.0])
    loss, frac = SL.uncertainty_guided_loss(p, ysl, 0.95)
    assert frac == 0.25 and abs(float(loss) - 2.0) < 1e-6


def test_selfloop_jigsaw_passes_leave_bn_running_stats_with_aux_bn_frozen():
    from cssl import selfloop as SL
    from model.cssl_net import CSSLNet
    from train import bn_stats_frozen, recalibrate_bn
    from contextlib import nullcontext
    torch.manual_seed(0)
    m = CSSLNet(3, 2, backbone="resunet18", pretrained=False, jigsaw_classes=100).train()
    pool = torch.from_numpy(SL.permutation_pool(100, seed=0))
    opt = torch.optim.SGD(list(m.unet.encoder.parameters()) + list(m.jigsaw.parameters()), lr=0.01)
    stats = lambda: {k: v.clone() for k, v in m.state_dict().items() if "running_" in k}
    before = stats()
    x, g = torch.rand(2, 3, 64, 64), torch.Generator().manual_seed(1)
    with bn_stats_frozen(m):
        SL.self_loop_pseudo_label(m, x, pool, 2, opt, g, nullcontext)
        SL.jigsaw_loss(m, x, pool, 2, g)[0].backward()
    assert all(torch.equal(v, before[k]) for k, v in stats().items())
    SL.jigsaw_loss(m, x, pool, 2, g)
    assert any(not torch.equal(v, before[k]) for k, v in stats().items() if "encoder" in k)
    data = [(torch.rand(3, 64, 64), torch.zeros(64, 64, dtype=torch.long))] * 4
    recalibrate_bn(m, data, data, 2, 2, torch.device("cpu"), num_workers=0)
    assert not m.training and all(b.momentum == 0.1 for b in m.modules() if isinstance(b, torch.nn.BatchNorm2d))


def test_monuseg_pooled_scores_match_confusion_matrix_definitions():
    from train import pooled_binary_scores
    rng = np.random.default_rng(0)
    gt = rng.random((3, 32, 32)) < 0.3
    pr = np.where(rng.random(gt.shape) < 0.8, gt, ~gt)
    tp, fp = int((pr & gt).sum()), int((pr & ~gt).sum())
    fn, tn = int((~pr & gt).sum()), int((~pr & ~gt).sum())
    d, f, b = pooled_binary_scores(tp, fp, fn, tn)
    prec, rec, spec = tp / (tp + fp), tp / (tp + fn), tn / (tn + fp)
    prec_bg, rec_bg = tn / (tn + fn), tn / (tn + fp)
    assert d == pytest.approx(2 * prec * rec / (prec + rec))
    assert f == pytest.approx((d + 2 * prec_bg * rec_bg / (prec_bg + rec_bg)) / 2)
    assert b == pytest.approx((rec + spec) / 2)


def test_feature_level_masking_embeddings():
    from model.cssl_net import CSSLNet
    torch.manual_seed(0)
    m = CSSLNet(3, 2, backbone="resunet18", pretrained=False, embed_level=-2).train()
    x = torch.rand(2, 3, 128, 128)
    lab = (torch.rand(2, 128, 128) < 0.3).long()
    P = sample_class_patches(x, lab, [0, 1], 64, 32, 25, 100, mode="each")
    assert P["boxes"].shape == (P["labels"].numel(), 3)
    z = m(x, "embed_image", masks=P["masks"], boxes=P["boxes"])
    assert z.shape == (P["labels"].numel(), 256) and torch.allclose(z.norm(dim=1), torch.ones(len(z)), atol=1e-5)
    b, y0, x0 = P["boxes"][0].tolist()
    f = m.unet.encoder(x)[-2]
    cells = f[b, :, y0 // 16:y0 // 16 + 4, x0 // 16:x0 // 16 + 4]
    w = F.avg_pool2d(P["masks"][0:1], 16)[0, 0]
    ref = F.normalize(m.proj((cells * w).sum((1, 2)) / w.sum()), dim=0)
    assert torch.allclose(z[0], ref, atol=1e-5)
    z.sum().backward()
    assert m.unet.encoder.layer3[0].conv1.weight.grad.abs().sum() > 0
    assert CSSLNet(3, 2, backbone="resunet18", pretrained=False).proj[0].in_features == 512


def test_memory_bank_adds_positives_and_negatives_without_gradient():
    from cssl.supcon import cssl_contrastive_loss
    torch.manual_seed(0)
    z = F.normalize(torch.randn(6, 8), dim=1).requires_grad_()
    lab = torch.tensor([0, 0, 0, 1, 1, 1])
    zm = F.normalize(torch.randn(10, 8), dim=1).requires_grad_()
    lm = torch.randint(0, 2, (10,))
    g = torch.Generator().manual_seed(0)
    base, _ = cssl_contrastive_loss(z, lab, hard_ratio=0.0, generator=g)
    with_mem, info = cssl_contrastive_loss(z, lab, hard_ratio=0.0, generator=g, memory=(zm, lm))
    assert info["cl_anchors"] == 6 and not torch.isclose(base, with_mem)
    with_mem.backward()
    assert z.grad is not None and zm.grad is None
    same, _ = cssl_contrastive_loss(z.detach(), lab, hard_ratio=0.0, generator=g, memory=(zm[:0], lm[:0]))
    assert torch.isclose(base.detach(), same)


def test_boundary_class_marks_the_nucleus_rim():
    lab = torch.zeros(1, 9, 9, dtype=torch.long)
    lab[0, 2:7, 2:7] = 1
    near_bg = F.max_pool2d((lab == 0).float()[:, None], 3, 1, 1)[:, 0] > 0
    out = torch.where((lab == 1) & near_bg, torch.full_like(lab, 2), lab)
    assert int((out == 2).sum()) == 16 and int((out == 1).sum()) == 9
    assert int((out == 0).sum()) == 81 - 25


def test_dafnet_rounding_matches_argmax_on_confident_masks():
    from train import dafnet_dice, dafnet_dice_masks
    g = torch.zeros(2, 8, 8, dtype=torch.long)
    g[0, :4, :4] = 1
    g[1, 4:, 4:] = 3
    p = g.clone()
    p[1, 4:6, 4:] = 0
    masks = F.one_hot(p, 5)[..., 1:].bool()
    assert dafnet_dice_masks(masks, g, 5) == pytest.approx(dafnet_dice(p, g, 5))
