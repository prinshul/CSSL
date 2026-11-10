import torch
import torch.nn.functional as F
import torchvision
from torch import nn

DEPTHS = {
    18: ("resnet18", "IMAGENET1K_V1", (64, 64, 128, 256, 512)),
    34: ("resnet34", "IMAGENET1K_V1", (64, 64, 128, 256, 512)),
    50: ("resnet50", "IMAGENET1K_V2", (64, 256, 512, 1024, 2048)),
}


def _conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResEncoder(nn.Module):
    def __init__(self, in_chns, pretrained=True, depth=18, input_norm="auto"):
        super().__init__()
        ctor, weights, self.channels = DEPTHS[depth]
        r = getattr(torchvision.models, ctor)(weights=weights if pretrained else None)
        if in_chns != 3:
            w = r.conv1.weight.data.mean(1, keepdim=True).repeat(1, in_chns, 1, 1)
            r.conv1 = nn.Conv2d(in_chns, 64, 7, 2, 3, bias=False)
            r.conv1.weight.data.copy_(w)
        if input_norm == "auto":
            input_norm = "imagenet" if in_chns == 3 else "mr"
        if input_norm not in ("imagenet", "mr"):
            raise ValueError("input_norm must be auto, imagenet or mr, not %r" % input_norm)
        self.mr_norm = input_norm == "mr"
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        if self.mr_norm:
            x = (x - 0.5) / 0.25
        f1 = self.stem(x)
        f2 = self.layer1(self.pool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return [f1, f2, f3, f4, f5]


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(_conv_bn_relu(cin + cskip, cout), _conv_bn_relu(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class ResDecoder(nn.Module):
    def __init__(self, nclass, enc_channels=(64, 64, 128, 256, 512), widths=(256, 128, 64, 64)):
        super().__init__()
        c1, c2, c3, c4, c5 = enc_channels
        w4, w3, w2, w1 = widths
        self.up4, self.up3 = _Up(c5, c4, w4), _Up(w4, c3, w3)
        self.up2, self.up1 = _Up(w3, c2, w2), _Up(w2, c1, w1)
        self.head = nn.Sequential(_conv_bn_relu(w1, 32), nn.Conv2d(32, nclass, 1))

    def forward(self, feature):
        f1, f2, f3, f4, f5 = feature
        x = self.up1(self.up2(self.up3(self.up4(f5, f4), f3), f2), f1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


class ResUNet(nn.Module):
    def __init__(self, in_chns, nclass, pretrained=True, depth=18, input_norm="auto"):
        super().__init__()
        self.encoder = ResEncoder(in_chns, pretrained, depth, input_norm)
        self.decoder = ResDecoder(nclass, self.encoder.channels)
        self.feat_dim = self.encoder.channels[-1]

    def forward(self, x, need_fp=False, perturb=None):
        feature = self.encoder(x)
        if perturb is not None:
            pert = perturb(feature)
            return self.decoder([torch.cat((f, g)) for f, g in zip(feature, pert)]).chunk(2)
        if need_fp:
            return self.decoder([torch.cat((f, nn.Dropout2d(0.5)(f))) for f in feature]).chunk(2)
        return self.decoder(feature)
