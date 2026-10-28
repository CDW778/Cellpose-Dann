#!/usr/bin/env python3
"""
ICPR2012 Path A 单细胞数据集生成（cell-on-white）。

- 读 cellpose_cache/{split}_masks_cp3 的 mask
- GT 区域分 1 连通块(普通) / >=2(末期)
- 正样本 = 普通 GT 的交集最大 mask
- 负样本 = 不与任何 GT 重叠的 mask，按 NEG_RATIO 采样
- 白板: mask disk(5) 膨胀 -> bbox -> 抠原图 -> mask内保留/外置纯白 -> 等比缩放居中 224
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import disk

SRC = Path("/home/hyw/dataset/chendawei/ICPR2012")
DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
BOARD = 224
FILL_FRAC = 0.9
DILATE = 5
NEG_RATIO = 4
SEED = 0
SPLITS = {"train": ["00", "01", "02"], "val": ["03"]}


def load_gt_regions(csv_p):
    regions = []
    for line in open(csv_p):
        line = line.strip()
        if not line:
            continue
        v = np.array([int(t) for t in line.split(",") if t != ""])
        regions.append((v[0::2], v[1::2]))
    return regions


def n_components(xs, ys):
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    m[ys - y0, xs - x0] = 1
    _, n = ndimage.label(m, structure=np.ones((3, 3)))
    return n


def find_csv(base):
    slide = base[:3]
    for d in SRC.glob("*_v2"):
        if d.name[:3] == slide:
            p = d / f"{base}.csv"
            if p.exists():
                return p
    return None


def label_bboxes(mask):
    """返回 {label: (y0,y1,x0,x1)}（stop 为开区间），用 find_objects 一次算出。"""
    sl = ndimage.find_objects(mask)
    out = {}
    for lab in np.unique(mask):
        lab = int(lab)
        if lab == 0:
            continue
        s = sl[lab - 1]
        out[lab] = (s[0].start, s[0].stop, s[1].start, s[1].stop)
    return out


def cell_on_white(img_bgr, mask, label, bbox=None):
    """mask==label 膨胀 disk(5) -> bbox -> 白底 -> 等比缩放居中到 BOARD。

    bbox 可选，给定 (y0,y1,x0,x1) 时只在该局部区域运算（快很多，结果一致）。
    """
    h, w = img_bgr.shape[:2]
    if bbox is None:
        ys, xs = np.where(mask == label)
        if len(ys) == 0:
            return None
        by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    else:
        by0, by1, bx0, bx1 = bbox
    # 外扩 DILATE 供膨胀使用，并限制在图内
    y0 = max(0, by0 - DILATE); y1 = min(h, by1 + DILATE)
    x0 = max(0, bx0 - DILATE); x1 = min(w, bx1 + DILATE)
    sub = mask[y0:y1, x0:x1]
    m = (sub == label).astype(np.uint8)
    if DILATE > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1)))
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return None
    yy0, yy1, xx0, xx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img_bgr[y0 + yy0:y0 + yy1, x0 + xx0:x0 + xx1]
    mcrop = m[yy0:yy1, xx0:xx1]
    white = np.full_like(crop, 255)
    white[mcrop > 0] = crop[mcrop > 0]
    ch, cw = white.shape[:2]
    scale = (FILL_FRAC * BOARD) / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(white, (nw, nh), interpolation=interp)
    board = np.full((BOARD, BOARD, 3), 255, np.uint8)
    ox, oy = (BOARD - nw) // 2, (BOARD - nh) // 2
    board[oy:oy + nh, ox:ox + nw] = resized
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=str(DST))
    ap.add_argument('--out-root', default=None, help='输出数据集根目录（默认=dst）')
    ap.add_argument('--neg-ratio', type=int, default=NEG_RATIO)
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED)

    dst = Path(args.dst)
    out_root = Path(args.out_root) if args.out_root else dst
    manifest = []
    # mean/std 累积（train）
    ssum = np.zeros(3, np.float64); ssq = np.zeros(3, np.float64); npx = 0

    for split, slides in SPLITS.items():
        img_dir = dst / "train-val-test" / f"{split}-original-img"
        mask_dir = dst / "cellpose_cache" / f"{split}_masks_cp3"
        pos_dir = out_root / split / "1_mitosis"
        neg_dir = out_root / split / "0_normal"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)

        for bmp in sorted(img_dir.glob("*.bmp")):
            base = bmp.stem
            mp = mask_dir / f"{base}_mask.npy"
            if not mp.exists():
                print(f"[跳过] 无 mask: {base}"); continue
            img = cv2.imread(str(bmp))
            mask = np.load(mp)
            csv_p = find_csv(base)
            if csv_p is None:
                print(f"[跳过] 无 GT: {base}"); continue
            regions = load_gt_regions(csv_p)

            gtmask = np.zeros(mask.shape, np.int32)
            comp_flags = []
            for i, (xs, ys) in enumerate(regions, 1):
                gtmask[ys, xs] = i
                comp_flags.append(n_components(xs, ys))

            gt_overlap = set(int(l) for l in np.unique(mask[gtmask > 0]) if l != 0)

            # 正样本：1 连通块 GT 的最大重叠 mask
            pos_labels = {}
            for i, (xs, ys) in enumerate(regions):
                if comp_flags[i] != 1:
                    continue  # 末期跳过
                labs, cnts = np.unique(mask[ys, xs], return_counts=True)
                nz = labs != 0
                if nz.sum() == 0:
                    manifest.append([split, base, i, "pos", "NO_OVERLAP", "", "", ""])
                    continue
                j = np.argmax(np.where(nz, cnts, 0))
                lab = int(labs[j])
                pos_labels.setdefault(lab, i)  # 同 mask 多 GT 去重

            # 负样本：不与任何 GT 重叠的 mask，按面积分 3 档均衡采样
            all_labels = [int(l) for l in np.unique(mask) if l != 0]
            neg_pool = [l for l in all_labels if l not in gt_overlap]
            n_neg = args.neg_ratio * len(pos_labels)
            if neg_pool:
                areas = ndimage.sum(np.ones(mask.shape, np.float32), mask, np.array(neg_pool))
                area_map = {l: float(a) for l, a in zip(neg_pool, areas)}
                neg_sorted = sorted(neg_pool, key=lambda l: area_map[l])
                if n_neg > 0 and len(neg_sorted) >= 3:
                    t = len(neg_sorted) // 3
                    tiers = [neg_sorted[:t], neg_sorted[t:2 * t], neg_sorted[2 * t:]]
                    per = n_neg // 3
                    neg_sel = []
                    for tier in tiers:
                        k = min(per, len(tier))
                        neg_sel += random.sample(tier, k)
                    rest = [l for l in neg_sorted if l not in set(neg_sel)]
                    need = n_neg - len(neg_sel)
                    if need > 0:
                        neg_sel += random.sample(rest, min(need, len(rest)))
                else:
                    random.shuffle(neg_sorted)
                    neg_sel = neg_sorted[:n_neg]
            else:
                neg_sel = []

            # 生成正样本
            for lab, gi in pos_labels.items():
                board = cell_on_white(img, mask, lab)
                if board is None:
                    continue
                fn = f"{base}_gt{gi}_pos.png"
                cv2.imwrite(str(pos_dir / fn), board)
                manifest.append([split, base, gi, "pos", lab, fn, "", ""])
                if split == "train":
                    b = board.astype(np.float64)
                    ssum += b.reshape(-1, 3).sum(0); ssq += (b.reshape(-1, 3) ** 2).sum(0); npx += b.shape[0] * b.shape[1]

            # 生成负样本
            for k, lab in enumerate(neg_sel):
                board = cell_on_white(img, mask, lab)
                if board is None:
                    continue
                fn = f"{base}_neg{k}.png"
                cv2.imwrite(str(neg_dir / fn), board)
                manifest.append([split, base, -1, "neg", lab, fn, "", ""])
                if split == "train":
                    b = board.astype(np.float64)
                    ssum += b.reshape(-1, 3).sum(0); ssq += (b.reshape(-1, 3) ** 2).sum(0); npx += b.shape[0] * b.shape[1]

            print(f"[{split}] {base}: pos={len(pos_labels)} neg={len(neg_sel)} gt_overlap={len(gt_overlap)}", flush=True)

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "hpf", "gt_idx", "class", "mask_label", "file", "note1", "note2"])
        w.writerows(manifest)

    if npx:
        mean = ssum / npx
        std = np.sqrt(np.maximum(ssq / npx - mean ** 2, 1e-6))
        print(f"\n[train 白板图 RGB 统计] mean={mean.tolist()} std={std.tolist()}  (BGR 顺序)")
    print(f"\n正样本数: train={len([m for m in manifest if m[0]=='train' and m[3]=='pos'])} "
          f"val={len([m for m in manifest if m[0]=='val' and m[3]=='pos'])}")
    print(f"负样本数: train={len([m for m in manifest if m[0]=='train' and m[3]=='neg'])} "
          f"val={len([m for m in manifest if m[0]=='val' and m[3]=='neg'])}")
    print("manifest:", out_root / "manifest.csv")


if __name__ == "__main__":
    main()
