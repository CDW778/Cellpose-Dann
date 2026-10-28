#!/usr/bin/env python3
"""DANN 微调 Dataset（RGB 原始窗口 + 可选 cellprob 第 4 通道 + 轻量/强增强）。"""
import numpy as np
import cv2
import torch
from pathlib import Path
from scipy import ndimage
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 224
_GLOBAL_CENT_CACHE = {}


def _bounds(cx, cy, size, H, W):
    half = size // 2
    l = max(0, cx - half); t = max(0, cy - half)
    r = l + size; b = t + size
    if r > W:
        r = W; l = r - size
    if b > H:
        b = H; t = b - size
    l = max(0, l); t = max(0, t)
    return l, t, r, b


def crop_patch(img_rgb, bbox, size=PATCH):
    H, W = img_rgb.shape[:2]
    left, top, right, bottom = bbox
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    l, t, r, b = _bounds(cx, cy, size, H, W)
    patch = img_rgb[t:b, l:r]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
    return patch


def crop_map(m, bbox, size=PATCH):
    H, W = m.shape[:2]
    left, top, right, bottom = bbox
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    l, t, r, b = _bounds(cx, cy, size, H, W)
    patch = m[t:b, l:r]
    patch = np.asarray(patch, dtype=np.float32)
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return patch


def crop_224(img_rgb, bbox):
    return crop_patch(img_rgb, bbox, PATCH)


def crop_at(arr, cx, cy, size=PATCH, interp=cv2.INTER_LINEAR):
    """以 (cx, cy) 为中心裁剪固定 size 窗口，越界内移。"""
    H, W = arr.shape[:2]
    l, t, r, b = _bounds(int(round(cx)), int(round(cy)), size, H, W)
    patch = arr[t:b, l:r]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=interp)
    return patch, (l, t, r, b)


def soft_mask_blend(patch, comp, sigma=4.0, dilate=4, bg_sigma=20.0, bg=None):
    """软 mask：目标内部保留原图，边界外与局部模糊背景融合。"""
    comp = (comp > 0).astype(np.uint8)
    if dilate > 0:
        k = 2 * int(dilate) + 1
        comp_d = cv2.dilate(comp, np.ones((k, k), np.uint8))
    else:
        comp_d = comp
    alpha = cv2.GaussianBlur(comp_d.astype(np.float32), (0, 0), float(sigma))
    alpha = np.maximum(alpha, comp.astype(np.float32))
    if bg is None:
        k = max(3, int(round(2.0 * float(bg_sigma))) | 1)
        bg = cv2.blur(patch, (k, k)).astype(np.float32)
    out = patch.astype(np.float32) * alpha[..., None] + bg * (1.0 - alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _shot_noise(p, scale=0.05):
    img = p.astype(np.float32)
    counts = img / 255.0 / scale
    noisy = np.random.poisson(counts).astype(np.float32) * scale * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _color_jitter(p):
    p = np.clip(p.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    mean = p.mean()
    p = np.clip((p - mean) * np.random.uniform(0.8, 1.2) + mean, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
    hsv[..., 0] = np.mod(hsv[..., 0] + np.random.uniform(-3, 3), 180)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _elastic(p, alpha=20.0, sigma=4.0):
    h, w = p.shape[:2]
    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(p, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _random_resized_crop(p, size):
    scale = np.random.uniform(0.7, 1.3)
    target = int(round(size * np.sqrt(scale)))
    if target > size:
        pad = (target - size) // 2 + 1
        pp = cv2.copyMakeBorder(p, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        yy = np.random.randint(0, pp.shape[0] - size + 1)
        xx = np.random.randint(0, pp.shape[1] - size + 1)
        return pp[yy:yy + size, xx:xx + size]
    yy = np.random.randint(0, size - target + 1)
    xx = np.random.randint(0, size - target + 1)
    c = p[yy:yy + target, xx:xx + target]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)


def light_augment(p):
    if np.random.rand() < 0.5:
        p = p[:, ::-1]
    if np.random.rand() < 0.5:
        p = p[::-1, :]
    k = np.random.randint(4)
    if k:
        p = np.rot90(p, k)
    return np.ascontiguousarray(p)


def strong_augment(p, size=PATCH):
    # 4 通道时仅做几何增强，避免颜色增强破坏 cellprob 通道
    p = light_augment(p)
    if p.shape[2] > 3:
        return np.ascontiguousarray(p)
    p = _random_resized_crop(p, size)
    ang = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
    p = cv2.warpAffine(p, M, (size, size), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    if np.random.rand() < 0.8:
        p = _color_jitter(p)
    if np.random.rand() < 0.3:
        p = cv2.GaussianBlur(p, (5, 5), np.random.uniform(0.3, 1.5))
    if np.random.rand() < 0.2:
        p = _shot_noise(p)
    if np.random.rand() < 0.15:
        p = cv2.GaussianBlur(p, (9, 9), np.random.uniform(1.5, 4.0))
    if np.random.rand() < 0.15:
        p = _elastic(p)
    return np.ascontiguousarray(p)


class CandidateDataset(Dataset):
    def __init__(self, split, cand, index, img_root, augment=False, strong=False,
                 crop_size=PATCH, in_chans=3, cellprob_dir=None, mask_dir=None,
                 crop_center='bbox', soft_mask=False, mask_sigma=4.0,
                 mask_dilate=4, mask_bg_sigma=20.0, mask_channel=False):
        self.split = split
        self.base = cand['base']; self.label = cand['label']
        self.x0 = cand['x0']; self.y0 = cand['y0']; self.x1 = cand['x1']; self.y1 = cand['y1']
        self.y = cand['y']; self.index = np.asarray(index)
        self.img_root = img_root
        self.augment = augment
        self.strong = strong
        self.crop_size = crop_size
        self.in_chans = in_chans
        self.cellprob_dir = Path(cellprob_dir) if cellprob_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.crop_center = crop_center
        self.soft_mask = bool(soft_mask)
        self.mask_channel = bool(mask_channel)
        self.mask_sigma = float(mask_sigma)
        self.mask_dilate = int(mask_dilate)
        self.mask_bg_sigma = float(mask_bg_sigma)
        if self.soft_mask and self.in_chans != 3:
            raise ValueError('soft_mask 目前只支持 in_chans=3')
        if self.soft_mask and self.mask_channel:
            raise ValueError('soft_mask 与 mask_channel 不能同时开启')
        if self.mask_channel and self.in_chans != 4:
            raise ValueError('mask_channel 需要 in_chans=4')
        if (self.soft_mask or self.mask_channel or self.crop_center == 'centroid') and self.mask_dir is None:
            raise ValueError('soft_mask / mask_channel / crop_center=centroid 需要提供 mask_dir')
        self._cache = {}
        self._cp_cache = {}
        self._mask_cache = {}
        self._cent_cache = {}
        if ((self.soft_mask or self.mask_channel or self.crop_center == 'centroid')
                and self.mask_dir is not None):
            for b in np.unique(self.base[self.index]):
                self._centroids(str(b))
            self._mask_cache.clear()

    def _img(self, base):
        if base not in self._cache:
            bgr = cv2.imread(str(self.img_root / f"{base}.bmp"))
            self._cache[base] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._cache[base]

    def _cellprob(self, base):
        if base not in self._cp_cache:
            p = Path(self.cellprob_dir) / f"{base}_cellprob.npy"
            self._cp_cache[base] = np.load(p)
        return self._cp_cache[base]

    def _mask(self, base):
        if base not in self._mask_cache:
            self._mask_cache[base] = np.load(self.mask_dir / f"{base}_mask.npy")
        return self._mask_cache[base]

    def _centroids(self, base):
        key = (str(self.mask_dir), base)
        if key not in _GLOBAL_CENT_CACHE:
            mask = self._mask(base)
            labels = np.unique(mask)
            labels = labels[labels != 0]
            cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
            _GLOBAL_CENT_CACHE[key] = {int(l): (float(c[0]), float(c[1]))
                                       for l, c in zip(labels, cents)}
        return _GLOBAL_CENT_CACHE[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        base = str(self.base[j])
        bbox = (int(self.x0[j]), int(self.y0[j]), int(self.x1[j]), int(self.y1[j]))
        img = self._img(base)
        size = self.crop_size
        win = None
        comp_c = None

        if self.soft_mask or self.mask_channel or self.crop_center == 'centroid':
            cent = self._centroids(base).get(int(self.label[j]))
            if self.crop_center == 'centroid' and cent is not None:
                cy, cx = cent
            else:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
            p, (l, t, r, b) = crop_at(img, cx, cy, size, cv2.INTER_CUBIC)
            if self.soft_mask or self.mask_channel:
                mask_c = self._mask(base)[t:b, l:r]
                comp_c = (mask_c == int(self.label[j])).astype(np.uint8)
                if comp_c.shape[0] != size or comp_c.shape[1] != size:
                    comp_c = cv2.resize(comp_c, (size, size), interpolation=cv2.INTER_NEAREST)
            if self.soft_mask:
                p = soft_mask_blend(p, comp_c, self.mask_sigma, self.mask_dilate,
                                    self.mask_bg_sigma)
            win = (l, t, r, b)
        else:
            p = crop_patch(img, bbox, size)

        if self.in_chans == 4:
            if self.mask_channel:
                p = np.concatenate([p, comp_c[..., None]], axis=2)
            else:
                if win is None:
                    cp = crop_map(self._cellprob(base), bbox, size)
                else:
                    l, t, r, b = win
                    cp = self._cellprob(base)[t:b, l:r].astype(np.float32)
                    if cp.shape[0] != size or cp.shape[1] != size:
                        cp = cv2.resize(cp, (size, size), interpolation=cv2.INTER_LINEAR)
                cp = np.clip(cp, -10.0, 10.0) / 5.0
                p = np.concatenate([p, cp[..., None]], axis=2)
        if self.augment:
            if self.strong and self.in_chans == 3:
                p = strong_augment(p, size)
            else:
                p = light_augment(p)
        if self.in_chans == 3:
            x = p.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        else:
            rgb = p[:, :, :3].astype(np.float32) / 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            cp = p[:, :, 3:4].astype(np.float32)
            x = np.concatenate([rgb, cp], axis=2)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()
        return x, int(self.y[j])
