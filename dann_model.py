#!/usr/bin/env python3
"""DANN 有丝分裂分类器（含 P0 核级注意力 / OD 门控 / 可选 cellprob 第 4 通道）。

基础结构来自原 final_model.py；本文件额外提供：
  - GeM 池化（参数无关）
  - CBAM（layer4 后）
  - OD/暗度门控（从 RGB 计算）
  - in_chans=4：额外 cellprob 通道
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

domain_attr = ['Tumor', 'Species', 'Origin', 'Scanner']


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(torch.tensor(lambda_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_val,) = ctx.saved_tensors
        return -lambda_val.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, val):
        self.lambda_val = val


class GeM(nn.Module):
    """广义平均池化：重点响应强激活（深染核）。"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ODGate(nn.Module):
    """用输入亮度/OD 生成空间门控，强调深染区域。"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feat, od):
        od = F.interpolate(od, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        g = torch.sigmoid(self.conv(od))
        return feat * (1.0 + self.alpha * g)


class DANNModel(nn.Module):
    def __init__(self, num_classes=2, num_domain_classes=None, lambda_val=0.0,
                 in_chans=3, use_gem=False, use_cbam=False, use_od_gate=False):
        super(DANNModel, self).__init__()
        self.in_chans = in_chans
        self.use_cbam = use_cbam
        self.use_od_gate = use_od_gate
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        backbone = models.resnet50(weights=None)
        conv1 = backbone.conv1
        if in_chans != 3:
            new_conv = nn.Conv2d(in_chans, conv1.out_channels,
                                 kernel_size=conv1.kernel_size,
                                 stride=conv1.stride,
                                 padding=conv1.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = conv1.weight
                if in_chans > 3:
                    new_conv.weight[:, 3:] = 0.0
            stem_conv = new_conv
        else:
            stem_conv = conv1

        self.stem = nn.Sequential(
            stem_conv,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.pool2 = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)
        self.pool3 = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)
        self.pool4 = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)

        self.cbam4 = CBAM(2048) if use_cbam else nn.Identity()
        self.od_gate4 = ODGate(2048) if use_od_gate else None

        feature_dim = 512 + 1024 + 2048

        self.mitosis_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes))

        self.grls = nn.ModuleDict({
            attr: GradientReversalLayer(lambda_val) for attr in domain_attr})

        self.domain_classifiers = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_domain_classes[attr]))
            for attr in domain_attr})

    def _od_map(self, x):
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        rgb = x[:, :3] * std + mean
        lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        return 1.0 - lum.clamp(0.0, 1.0)

    def _features(self, x):
        od = self._od_map(x) if self.use_od_gate else None
        x = self.stem(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.use_cbam:
            x4 = self.cbam4(x4)
        if self.use_od_gate:
            x4 = self.od_gate4(x4, od)
        f2 = self.pool2(x2).flatten(1)
        f3 = self.pool3(x3).flatten(1)
        f4 = self.pool4(x4).flatten(1)
        return torch.cat([f2, f3, f4], dim=1)

    def forward(self, x):
        features = self._features(x)
        mitosis_logits = self.mitosis_classifier(features)
        domain_logits = {}
        for attr in domain_attr:
            reversed_features = self.grls[attr](features)
            domain_logits[attr] = self.domain_classifiers[attr](reversed_features)
        return mitosis_logits, domain_logits

    def set_lambda(self, val):
        for attr in domain_attr:
            self.grls[attr].set_lambda(val)

    def predict_only(self, x):
        return self.mitosis_classifier(self._features(x))

    def get_features(self, x):
        with torch.no_grad():
            return self._features(x)


def load_pretrained_dann(model, checkpoint):
    """加载 3 通道预训练权重；若 in_chans>3，第 4 通道保持初始化(0)。"""
    sd = torch.load(checkpoint, map_location='cpu')
    key = 'stem.0.weight'
    if key in sd and sd[key].shape != model.stem[0].weight.shape:
        w = sd[key]
        new = model.stem[0].weight.detach().clone()
        new[:, :w.shape[1]] = w
        sd[key] = new
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected
