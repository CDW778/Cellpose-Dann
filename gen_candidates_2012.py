#!/usr/bin/env python3
"""为 train/val 生成 Cellpose 候选索引（供 DANN 二阶段微调）。

正样本 = 普通(1 连通块) GT 区域的最大重叠 mask；负样本 = 非 GT 重叠 mask；
排除与 GT 重叠但非最大的歧义 mask；末期(>=2 连通块)不作正。
"""
import argparse
import numpy as np
from pathlib import Path
from scipy import ndimage

from gen_dataset_2012 import find_csv, load_gt_regions, label_bboxes, n_components

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT_DEFAULT = "/home/hyw/chendawei/dann_finetune"
SPLIT_SLIDES = {
    "train": {"A00", "A01", "A02", "H00", "H01", "H02"},
    "val": {"A03", "H03"},
}


def build(split, slides):
    img_dir = DST / "train-val-test" / f"{split}-original-img"
    mask_dir = DST / "cellpose_cache" / f"{split}_masks_cp3"
    rows = []
    npos = nneg = 0
    for bmp in sorted(img_dir.glob("*.bmp")):
        base = bmp.stem
        if base[:3] not in slides:
            continue
        mask = np.load(mask_dir / f"{base}_mask.npy")
        labels = [int(l) for l in np.unique(mask) if l != 0]
        bboxes = label_bboxes(mask)
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        coord = {int(l): (float(c[1]), float(c[0])) for l, c in zip(labels, cents)}
        csv_p = find_csv(base)
        if csv_p is None:
            continue
        regions = load_gt_regions(csv_p)

        gtmask = np.zeros(mask.shape, np.int32)
        for i, (xs, ys) in enumerate(regions, 1):
            gtmask[ys, xs] = i
        gt_overlap = set(int(l) for l in np.unique(mask[gtmask > 0]) if l != 0)

        pos = set()
        for xs, ys in regions:
            if n_components(xs, ys) != 1:
                continue
            labs, cnts = np.unique(mask[ys, xs], return_counts=True)
            nz = labs != 0
            if nz.sum() == 0:
                continue
            pos.add(int(labs[np.argmax(np.where(nz, cnts, 0))]))

        for lab in labels:
            if lab in pos:
                y = 1
            elif lab in gt_overlap:
                continue  # 歧义 mask 不入负样本
            else:
                y = 0
            y0, y1, x0, x1 = bboxes[lab]
            cx, cy = coord[lab]
            rows.append((base, lab, x0, y0, x1, y1, y, cx, cy))
            npos += (y == 1)
            nneg += (y == 0)
        print(f"[{split}] {base}: pos+={len(pos)} gt_ov={len(gt_overlap)}", flush=True)
    bases = np.array([r[0] for r in rows])
    labs = np.array([r[1] for r in rows], np.int64)
    x0 = np.array([r[2] for r in rows], np.int32)
    y0 = np.array([r[3] for r in rows], np.int32)
    x1 = np.array([r[4] for r in rows], np.int32)
    y1 = np.array([r[5] for r in rows], np.int32)
    y = np.array([r[6] for r in rows], np.int8)
    cx = np.array([r[7] for r in rows], np.float32)
    cy = np.array([r[8] for r in rows], np.float32)
    print(f"[{split}] 候选={len(rows)} 正={int(y.sum())} 负={int((y==0).sum())}", flush=True)
    return dict(base=bases, label=labs, x0=x0, y0=y0, x1=x1, y1=y1, y=y, cx=cx, cy=cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for split, slides in SPLIT_SLIDES.items():
        data = build(split, slides)
        p = out / f"candidates_{split}.npz"
        np.savez_compressed(p, **data)
        print(f"写出: {p}", flush=True)


if __name__ == "__main__":
    main()
