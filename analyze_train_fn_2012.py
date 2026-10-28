#!/usr/bin/env python3
"""拆解 train(00-02) 的 FN：候选/匹配/分割型 vs 分类型。

用 best_s3_seed0.pth 给 train 全部 Cellpose mask 打分，按 5µm 贪心匹配；
在 oracle 阈值（默认 0.99）下找出 FN，并统计每个 FN：
  - 5µm 内是否有 mask、最高分、最近距离
  - GT 连通块数（>=2 末期）、GT 面积、mask 面积、最大 IoU
分类：
  - no_mask_5um      : 5µm 内没有 mask -> 候选/分割/匹配
  - matching_conflict: 5µm 内有 mask 且最高分>=thr，但未匹配（被其他 GT 占用）
  - low_score        : 5µm 内有 mask，但最高分<thr -> 分类漏检
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage

from dann_model import DANNModel
from dann_dataset import crop_patch
from gen_dataset_2012 import find_csv, load_gt_regions, label_bboxes, n_components
from eval_2012_dann_live import res_um, load_gt_points

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT = Path("/home/hyw/chendawei/dann_finetune")
DANN_WEIGHT = str(OUT / "best_s3_seed0.pth")
NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 224
BATCH = 256
DIST_UM = 5.0
THR = 0.99


def score_train(device, use_cache=True):
    cache = OUT / "train_scores_ce_seed0.npz"
    if use_cache and cache.exists():
        print(f"读取缓存 {cache}", flush=True)
        return np.load(cache, allow_pickle=True)
    model = DANNModel(2, NDC, 0.0)
    model.load_state_dict(torch.load(DANN_WEIGHT, map_location="cpu"))
    model.eval().to(device)
    img_dir = DST / "train-val-test" / "train-original-img"
    mask_dir = DST / "cellpose_cache" / "train_masks_cp3"
    rows = []
    bmps = sorted(img_dir.glob("*.bmp"))
    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_dir / f"{base}_mask.npy")
        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        bb = label_bboxes(mask)
        for i in range(0, len(labels), BATCH):
            chunk = labels[i:i + BATCH]
            patches = []
            for lab in chunk:
                y0, y1, x0, x1 = bb[lab]
                patches.append(crop_patch(img, (x0, y0, x1, y1), PATCH))
            tens = []
            for p in patches:
                x = p.astype(np.float32) / 255.0
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
                tens.append(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float())
            with torch.no_grad():
                prob = torch.softmax(model.predict_only(torch.stack(tens).to(device)), 1)[:, 1].cpu().numpy()
            for k, lab in enumerate(chunk):
                cy, cx = cents[i + k]
                rows.append((base, lab, float(cx), float(cy), float(prob[k])))
        print(f"[{hi}/{len(bmps)}] {base}: masks={len(labels)}", flush=True)
    base_a = np.array([r[0] for r in rows])
    lab_a = np.array([r[1] for r in rows], np.int64)
    cx_a = np.array([r[2] for r in rows], np.float32)
    cy_a = np.array([r[3] for r in rows], np.float32)
    p_a = np.array([r[4] for r in rows], np.float32)
    np.savez_compressed(cache, base=base_a, label=lab_a, cx=cx_a, cy=cy_a, p=p_a)
    print(f"写出缓存 {cache} n={len(rows)}", flush=True)
    return np.load(cache, allow_pickle=True)


def match(dets, gts, thr_px):
    """dets: [(x,y,p,label)], gts: [(x,y)]; greedy one-to-one."""
    used = set(); matched = set(); pairs = []
    for gi, (gx, gy) in enumerate(gts):
        best = None
        for di, (x, y, p, lab) in enumerate(dets):
            if di in used:
                continue
            d = float(np.hypot(x - gx, y - gy))
            if d <= thr_px and (best is None or d < best[0]):
                best = (d, di)
        if best:
            used.add(best[1]); matched.add(gi); pairs.append((gi, best[1]))
    return pairs, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=float, default=THR)
    ap.add_argument('--dist-um', type=float, default=DIST_UM)
    ap.add_argument('--no-cache', dest='use_cache', action='store_false')
    ap.add_argument('--out', default=str(OUT / "train_fn_breakdown.csv"))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}; thr={args.thr}; dist={args.dist_um}um", flush=True)
    d = score_train(device, args.use_cache)

    # group scores by image
    by = {}
    for i in range(len(d['base'])):
        b = str(d['base'][i])
        by.setdefault(b, []).append((float(d['cx'][i]), float(d['cy'][i]),
                                     float(d['p'][i]), int(d['label'][i])))

    rows = []
    tp = fp = fn = 0
    total_gt = 0
    for base in sorted(by):
        recs = by[base]
        gts, _ = load_gt_points(find_csv(base))
        thr_px = args.dist_um / res_um(base)
        dets = [(x, y, p, lab) for (x, y, p, lab) in recs if p >= args.thr]
        pairs, matched = match(dets, gts, thr_px)
        tp += len(pairs); fp += len(dets) - len(pairs); fn += len(gts) - len(matched)
        total_gt += len(gts)
        regions = load_gt_regions(find_csv(base))
        for gi, (gx, gy) in enumerate(gts):
            if gi in matched:
                continue
            xs, ys = regions[gi]; nc = n_components(xs, ys); gta = len(xs)
            # all masks within 5um (any score)
            near = []
            for (x, y, p, lab) in recs:
                dd = float(np.hypot(x - gx, y - gy))
                if dd <= thr_px:
                    near.append((dd, x, y, p, lab))
            if not near:
                best_score = -1.0; nearest_dist = -1.0; nearest_score = -1.0
                best_iou = 0.0; mask_area = -1; cat = "no_mask_5um"
            else:
                near.sort(key=lambda t: t[0])
                nearest_dist, _, _, nearest_score, nearest_lab = near[0]
                best_score = max(t[3] for t in near)
                # best IoU among near masks
                mask_area = 0; best_iou = 0.0
                mask_full = np.load(DST / "cellpose_cache" / "train_masks_cp3" / f"{base}_mask.npy")
                for (_, _, _, _, lab) in near:
                    inter = int(np.sum(mask_full[ys, xs] == lab))
                    ma = int(np.sum(mask_full == lab))
                    iou = inter / (gta + ma - inter) if (gta + ma - inter) > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou; mask_area = ma
                cat = "matching_conflict" if best_score >= args.thr else "low_score"
            rows.append(dict(base=base, gt=gi, ncomp=nc, gt_area=gta,
                             n_near=len(near), best_score=round(best_score, 4),
                             nearest_dist_um=round(nearest_dist * res_um(base), 2) if nearest_dist >= 0 else -1,
                             nearest_score=round(nearest_score, 4),
                             mask_area=mask_area, best_iou=round(best_iou, 3), category=cat))

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nGT={total_gt} TP={tp} FP={fp} FN={fn} (thr={args.thr})", flush=True)
    print(f"FN={len(rows)}", flush=True)
    from collections import Counter
    print("categories:", dict(Counter(r['category'] for r in rows)), flush=True)
    print("FN 中 ncomp>=2:", sum(1 for r in rows if r['ncomp'] >= 2), flush=True)
    for cat in ["no_mask_5um", "low_score", "matching_conflict"]:
        sub = [r for r in rows if r['category'] == cat]
        if not sub:
            continue
        print(f"\n[{cat}] n={len(sub)}", flush=True)
        print("  best_score:", [r['best_score'] for r in sub][:20], flush=True)
        print("  nearest_dist_um:", [r['nearest_dist_um'] for r in sub][:20], flush=True)
        print("  best_iou:", [r['best_iou'] for r in sub][:20], flush=True)
    print(f"\n写出: {args.out}", flush=True)


if __name__ == '__main__':
    main()
