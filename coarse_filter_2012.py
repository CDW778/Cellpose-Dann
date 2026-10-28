#!/usr/bin/env python3
"""ICPR2012 几何粗筛：用 Cellpose mask 的几何先验硬过滤明显碎屑/巨团块/细长杂质。

校准口径（只用 train 00-02，严格保留全部 e2e 匹配到的 GT mask）：
  1) 在 train 上用 5µm 一对一贪心匹配，取出被匹配到 GT 的 mask（TP masks）；
  2) 每条规则阈值取"包含全部 TP"的边界，外扩 MARGIN；
  3) 输出 filter_geom.json（仓库外）+ 各规则削减率与 GT 存活率报告。
"""
import argparse
import json
from pathlib import Path
from math import pi

import numpy as np
from skimage.measure import regionprops

from eval_2012_dann_live import res_um, load_gt_points, match, load_masks_from_cache
from gen_dataset_2012 import find_csv

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT_DEFAULT = "/home/hyw/chendawei/dann_finetune/filter_geom.json"
DIST_UM = 5.0
MARGIN = 0.05
SPLITS = {"train": ["00", "01", "02"], "val": ["03"]}
TEST_SLIDES = ["04"]


def compute_features(mask):
    """返回 {label: dict(area,equiv_diam,solidity,circularity,aspect,eccentricity)}。"""
    feats = {}
    for r in regionprops(mask.astype(np.int32)):
        area = float(r.area)
        per = float(r.perimeter)
        minr, minc, maxr, maxc = r.bbox
        h = max(maxr - minr, 1)
        w = max(maxc - minc, 1)
        feats[int(r.label)] = {
            'area': area,
            'equiv_diam': float(r.equivalent_diameter),
            'solidity': float(r.solidity),
            'circularity': (4.0 * pi * area / (per * per)) if per > 0 else 0.0,
            'aspect': float(w) / float(h),
            'eccentricity': float(r.eccentricity),
        }
    return feats


def passes(feat, thr):
    if 'area' in thr:
        lo, hi = thr['area']
        if feat['area'] < lo or feat['area'] > hi:
            return False
    if feat['solidity'] < thr.get('solidity_min', -1):
        return False
    if feat['circularity'] < thr.get('circularity_min', -1):
        return False
    if feat['aspect'] > thr.get('aspect_max', 1e9):
        return False
    if feat['eccentricity'] > thr.get('eccentricity_max', 1e9):
        return False
    return True


def collect(split):
    """返回 list of (base, mask, feats, gts, thr_px, centroids)。"""
    dst = DST
    img_dir = dst / "train-val-test" / f"{split}-original-img"
    items = []
    if not img_dir.exists():
        return items
    for bmp in sorted(img_dir.glob("*.bmp")):
        base = bmp.stem
        mask = load_masks_from_cache(dst, base) if split == "test" else np.load(
            dst / "cellpose_cache" / f"{split}_masks_cp3" / f"{base}_mask.npy")
        if mask is None:
            continue
        from scipy import ndimage
        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        coord = {int(l): (float(c[1]), float(c[0])) for l, c in zip(labels, cents)}
        feats = compute_features(mask)
        csv_p = find_csv(base)
        gts = load_gt_points(csv_p) if csv_p else []
        items.append((base, mask, feats, gts, DIST_UM / res_um(base), coord))
    return items


def tp_masks(items):
    """在 train 上跑无阈值匹配，返回被匹配的 (label) 集合特征。"""
    tp_feats = []
    for (base, mask, feats, gts, thr_px, coord) in items:
        dets = [coord[l] for l in feats]
        dets_xy = np.array(dets) if dets else np.zeros((0, 2))
        matched_d = set()
        for gx, gy in gts:
            if len(dets_xy) == 0:
                break
            dd = np.sqrt((dets_xy[:, 0] - gx) ** 2 + (dets_xy[:, 1] - gy) ** 2)
            for di in np.argsort(dd):
                if dd[di] > thr_px:
                    break
                if di in matched_d:
                    continue
                matched_d.add(int(di)); break
        labs = list(feats.keys())
        for di in matched_d:
            tp_feats.append(feats[labs[di]])
    return tp_feats


def calibrate(tp_feats):
    arr = {k: np.array([f[k] for f in tp_feats]) for k in
           ['area', 'equiv_diam', 'solidity', 'circularity', 'aspect', 'eccentricity']}
    m = MARGIN
    thr = {
        'area': [float(arr['area'].min() * (1 - m)), float(arr['area'].max() * (1 + m))],
        'solidity_min': float(arr['solidity'].min() * (1 - m)),
        'circularity_min': float(arr['circularity'].min() * (1 - m)),
        'aspect_max': float(arr['aspect'].max() * (1 + m)),
        'eccentricity_max': float(arr['eccentricity'].max() * (1 + m)),
    }
    return thr, arr


def survival(items, thr):
    tot_gt = 0; survive_gt = 0; total_masks = 0; kept_masks = 0
    for (base, mask, feats, gts, thr_px, coord) in items:
        total_masks += len(feats)
        keep = [l for l in feats if passes(feats[l], thr)]
        kept_masks += len(keep)
        dets = [(coord[l][0], coord[l][1], 1.0) for l in keep]
        t, _, _ = match(dets, gts, thr_px)
        tot_gt += len(gts); survive_gt += t
    return tot_gt, survive_gt, total_masks, kept_masks


def rule_report(items, thr):
    """每条规则单独在 train 上的削减率。"""
    rules = ['area', 'solidity_min', 'circularity_min', 'aspect_max', 'eccentricity_max']
    out = {}
    for rk in rules:
        keep = 0; tot = 0
        for (base, mask, feats, gts, thr_px, coord) in items:
            for l, f in feats.items():
                tot += 1
                sub = {rk: thr[rk]}
                if passes(f, sub):
                    keep += 1
        out[rk] = 1.0 - keep / max(tot, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_DEFAULT)
    args = ap.parse_args()

    print("收集 train/val/test ...", flush=True)
    train = collect("train"); val = collect("val"); test = collect("test")
    print(f"train imgs={len(train)} val={len(val)} test={len(test)}", flush=True)

    tps = tp_masks(train)
    print(f"train TP masks = {len(tps)}", flush=True)
    thr, arr = calibrate(tps)
    print("\n=== 校准阈值（含 5% margin，保留全部 train TP）===")
    print(json.dumps(thr, indent=2, ensure_ascii=False))
    print("\n=== 各规则单独的候选削减率（train）===")
    print(json.dumps(rule_report(train, thr), indent=2, ensure_ascii=False))

    print("\n=== 组合过滤后（train / val / test）===")
    for name, items in [("train", train), ("val", val), ("test", test)]:
        tgt, sgt, tm, km = survival(items, thr)
        print(f"{name:5s}: GT {sgt}/{tgt} 存活  | 候选 {km}/{tm} 保留 (削减 {100*(1-km/max(tm,1)):.1f}%)",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({'features_thresholds': thr, 'margin': MARGIN}, open(args.out, 'w'), indent=2)
    print(f"\n写出: {args.out}")


if __name__ == "__main__":
    main()
