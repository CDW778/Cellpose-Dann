#!/usr/bin/env python3
"""审计：显示 4 个 FN 案例里 Cellpose 分割出的所有 mask，以及相关 224 裁剪。"""
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

from gen_dataset_2012 import find_csv, load_gt_regions, label_bboxes, n_components

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT = Path("/home/hyw/chendawei/dann_finetune/mask_audit")
PATCH = 224
CASES = [("A04_02", 3, 4), ("A04_04", 4, 1), ("H04_04", 1, 4), ("H04_04", 4, 1)]


def gt_bin(xs, ys, shape):
    b = np.zeros(shape, np.uint8); b[ys, xs] = 1; return b


def draw_gt(ann, binm, color, thick=2):
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, color, thick)


def full_overlay(img, mask, tgt_lab, extra_labs, tgt_gt, extra_gt, crop_box):
    rng = np.random.RandomState(123)
    colors = rng.randint(30, 255, (int(mask.max()) + 1, 3), dtype=np.uint8)
    colors[0] = (0, 0, 0)
    overlay = img.copy()
    m = mask > 0
    overlay[m] = (img[m] * 0.5 + colors[mask][m] * 0.5).astype(np.uint8)
    b = find_boundaries(mask, mode="outer")
    overlay[b] = (255, 255, 255)
    # mask contours
    if tgt_lab > 0:
        draw_gt(overlay, (mask == tgt_lab).astype(np.uint8), (0, 0, 255), 3)
    for l in extra_labs:
        if l > 0:
            draw_gt(overlay, (mask == l).astype(np.uint8), (0, 165, 255), 3)
    # GT regions
    draw_gt(overlay, tgt_gt, (0, 255, 0), 3)
    draw_gt(overlay, extra_gt, (255, 0, 255), 3)
    # 224 crop box
    x0, y0, x1, y1 = crop_box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 3)
    return overlay


def crop_224(img, cx, cy):
    H, W = img.shape[:2]
    l = max(0, cx - PATCH // 2); t = max(0, cy - PATCH // 2)
    r = l + PATCH; b = t + PATCH
    if r > W: r = W; l = r - PATCH
    if b > H: b = H; t = b - PATCH
    l = max(0, l); t = max(0, t)
    return img[t:b, l:r].copy(), (l, t, l + PATCH, t + PATCH)


def crop_overlay_from_full(full, box, tgt_lab, extra_labs, tgt_gt, extra_gt, mask):
    x0, y0, x1, y1 = box
    c = full[y0:y1, x0:x1].copy()
    # 在裁剪内重画，避免线条被裁断
    if tgt_lab > 0:
        draw_gt(c, (mask[y0:y1, x0:x1] == tgt_lab).astype(np.uint8), (0, 0, 255), 2)
    for l in extra_labs:
        if l > 0:
            draw_gt(c, (mask[y0:y1, x0:x1] == l).astype(np.uint8), (0, 165, 255), 2)
    draw_gt(c, tgt_gt[y0:y1, x0:x1], (0, 255, 0), 2)
    draw_gt(c, extra_gt[y0:y1, x0:x1], (255, 0, 255), 2)
    return c


def tile(img, title, size):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((size, size))
    canvas = Image.new("RGB", (size, size + 24), (255, 255, 255))
    canvas.paste(im, (0, 24))
    ImageDraw.Draw(canvas).text((4, 6), title, fill=(0, 0, 0))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles_all = []
    for base, tgt_i, extra_i in CASES:
        img = cv2.imread(str(DST / "train-val-test" / "test-original-img" / f"{base}.bmp"))
        mask = np.load(DST / "cellpose_cache" / "test_masks_cp3" / f"{base}_mask.npy")
        bb = label_bboxes(mask)
        regs = load_gt_regions(find_csv(base))
        xs_t, ys_t = regs[tgt_i]; xs_e, ys_e = regs[extra_i]
        # target mask = max IoU
        bt = (0, -1)
        for l in [int(v) for v in np.unique(mask[ys_t, xs_t]) if v != 0]:
            inter = int(np.sum(mask[ys_t, xs_t] == l)); ma = int(np.sum(mask == l))
            iou = inter / (len(xs_t) + ma - inter)
            if iou > bt[0]: bt = (iou, l)
        tgt_lab = bt[1] if bt[1] > 0 else int(np.unique(mask[ys_t, xs_t])[0])
        # extra GT 覆盖 mask：取交集最大的
        labs_e = [int(v) for v in np.unique(mask[ys_e, xs_e]) if v != 0]
        extra_labs = []
        if labs_e:
            best = max(labs_e, key=lambda l: int(np.sum(mask[ys_e, xs_e] == l)))
            extra_labs = [best]
        # 224 crop around target mask bbox center
        y0, y1, x0, x1 = bb[tgt_lab]
        bcx, bcy = (x0 + x1) // 2, (y0 + y1) // 2
        raw224, box224 = crop_224(img, bcx, bcy)
        # 224 crop around extra mask bbox center
        if extra_labs:
            ey0, ey1, ex0, ex1 = bb[extra_labs[0]]
            raw224e, box224e = crop_224(img, (ex0 + ex1) // 2, (ey0 + ey1) // 2)
        else:
            raw224e, box224e = raw224, box224

        tgt_gt = gt_bin(xs_t, ys_t, mask.shape)
        extra_gt = gt_bin(xs_e, ys_e, mask.shape)
        full = full_overlay(img, mask, tgt_lab, extra_labs, tgt_gt, extra_gt, box224)

        # 全图（缩放到 700）
        full_small = cv2.resize(full, (700, 700), interpolation=cv2.INTER_AREA)
        # zoom：target + extra 的联合 bbox + 120 margin
        ux0 = min(xs_t.min(), xs_e.min()); ux1 = max(xs_t.max(), xs_e.max())
        uy0 = min(ys_t.min(), ys_e.min()); uy1 = max(ys_t.max(), ys_e.max())
        m = 120
        zx0 = max(0, ux0 - m); zy0 = max(0, uy0 - m)
        zx1 = min(img.shape[1], ux1 + m); zy1 = min(img.shape[0], uy1 + m)
        zoom = full[zy0:zy1, zx0:zx1].copy()
        zoom = cv2.resize(zoom, (560, 560), interpolation=cv2.INTER_AREA)
        # 224 crops（原始 + 所有 mask 边界 + GT）
        c_t = crop_overlay_from_full(full, box224, tgt_lab, extra_labs, tgt_gt, extra_gt, mask)
        c_e = crop_overlay_from_full(full, box224e, tgt_lab, extra_labs, tgt_gt, extra_gt, mask)

        top = Image.new("RGB", (700 + 560, 724), (255, 255, 255))
        top.paste(tile(full_small, f"{base}: ALL Cellpose masks (red=target mask, orange=extra-GT mask, green=target GT, magenta=extra GT, yellow=224 box)", 700), (0, 0))
        top.paste(tile(zoom, "zoom: target+extra", 560), (700, 0))
        bot = Image.new("RGB", (448, 248), (255, 255, 255))
        bot.paste(tile(c_t, f"224 crop @ target mask ({tgt_lab})", 224), (0, 0))
        bot.paste(tile(c_e, f"224 crop @ extra-GT mask ({extra_labs[0] if extra_labs else '-'})", 224), (224, 0))
        combo = Image.new("RGB", (max(700 + 560, 448), 724 + 248), (255, 255, 255))
        combo.paste(top, (0, 0)); combo.paste(bot, (0, 724))
        fn = OUT / f"{base}_gt{tgt_i}_extra{extra_i}_maskaudit.png"
        combo.save(fn)
        tiles_all.append(str(fn))
        print(f"{fn}  target_mask={tgt_lab} extra_mask={extra_labs} extra_gt={extra_i}", flush=True)

    # 4 个案例总览
    if tiles_all:
        ims = [Image.open(p) for p in tiles_all]
        w = max(i.width for i in ims); h = sum(i.height for i in ims)
        sheet = Image.new("RGB", (w, h), (255, 255, 255)); y = 0
        for im in ims:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(OUT / "all_cases_maskaudit.png")
        print(f"总览: {OUT/'all_cases_maskaudit.png'}")


if __name__ == "__main__":
    main()
