import torch
import torch.nn.functional as F
from torch import nn

from .resunet import ResUNet
from .unet_unimatch import UNet

_BN = nn.modules.batchnorm._BatchNorm

RESUNETS = {"resunet18": 18, "resunet34": 34, "resunet50": 50}


class CSSLNet(nn.Module):
    def __init__(self, in_chns, nclass, widths=(32, 64, 128, 256, 512), embed_dim=256,
                 backbone="unet", pretrained=True, input_norm="auto", jigsaw_classes=0, jigsaw_hidden=512,
                 embed_level=-1):
        super().__init__()
        self.pm1 = input_norm == "pm1"
        if self.pm1:
            input_norm = "imagenet"
        if backbone in RESUNETS:
            self.unet = ResUNet(in_chns, nclass, pretrained=pretrained, depth=RESUNETS[backbone], input_norm=input_norm)
            chans = list(self.unet.encoder.channels)
        elif backbone == "unet":
            self.unet, chans = UNet(in_chns, nclass, feature_chns=widths), list(widths)
        else:
            raise ValueError(backbone)
        self.embed_level = int(embed_level)
        feat_dim = chans[self.embed_level]
        self.proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
                                  nn.Linear(feat_dim, embed_dim))
        self.jigsaw = None
        if jigsaw_classes:
            from cssl.selfloop import JigsawHead
            self.jigsaw = JigsawHead(chans[-1], jigsaw_classes, jigsaw_hidden)

    def forward(self, x, mode="seg", masks=None, perturb=None, boxes=None):
        if self.pm1:
            x = x * 2.0 - 1.0
        if mode == "seg":
            return self.unet(x)
        if mode == "seg_cr":
            return self.unet(x, perturb=perturb)
        if mode == "seg_fp":
            return self.unet(x, need_fp=True)
        if mode == "jigsaw_only":
            return self.jigsaw(self.unet.encoder(x)[-1])
        if mode == "jigsaw":
            f = self.unet.encoder(x)
            logits = self.jigsaw(f[-1])
            bns = [m for m in self.unet.decoder.modules() if isinstance(m, _BN)]
            saved = [m.momentum for m in bns]
            for m in bns:
                m.momentum = 0.0
            try:
                with torch.no_grad():
                    seg = self.unet.decoder([t.detach() for t in f])
            finally:
                for m, mom in zip(bns, saved):
                    m.momentum = mom
            return logits, seg
        if mode == "embed_image":
            f = self.unet.encoder(x)[self.embed_level].float()
            stride, s = x.shape[-1] // f.shape[-1], masks.shape[-1]
            if s % stride or bool((boxes[:, 1:] % stride).any()):
                raise ValueError("patch windows must lie on the feature grid (stride %d)" % stride)
            cs, ar = s // stride, torch.arange(s // stride, device=f.device)
            rows, cols = boxes[:, 1:2] // stride + ar, boxes[:, 2:3] // stride + ar
            crops = f[boxes[:, 0, None, None], :, rows[:, :, None], cols[:, None, :]]
            m = F.interpolate(masks.float(), size=(cs, cs), mode="area")[:, 0, :, :, None]
            den = m.sum((1, 2))
            pooled = (crops * m).sum((1, 2)) / den.clamp(min=1e-6)
            pooled = torch.where(den > 1e-6, pooled, crops.mean((1, 2)))
            return F.normalize(self.proj(pooled), dim=1)
        if mode == "embed":
            f = self.unet.encoder(x)[self.embed_level]
            m = F.interpolate(masks.float(), size=f.shape[-2:], mode="area")
            den = m.sum((2, 3))
            pooled = (f.float() * m).sum((2, 3)) / den.clamp(min=1e-6)
            pooled = torch.where(den > 1e-6, pooled, f.float().mean((2, 3)))
            return F.normalize(self.proj(pooled), dim=1)
        raise ValueError(mode)
