#!/usr/bin/env python3
"""ICPR2012 端到端有丝分裂检测评估（DANN 版本）。

流程：Cellpose 分割(现场或缓存) -> 每个 mask 以 bbox 中心裁 224 原始 RGB 窗口
      -> DANN 打分 -> mask 质心为检测点 -> GT 5µm 一对一贪心匹配 -> F1。

与 eval_2012_live.py 的差异：二阶段分类器换成 DANN(ResNet50+域对抗, MIDOG++)，
输入换成 224 原始 RGB 窗口(ImageNet 归一化)，不读 ConvNeXt config/权重。

用法:
  python eval_2012_dann_live.py                 # 用缓存 mask
  python eval_2012_dann_live.py --live          # 现场 Cellpose 推理
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage

from dann_model import DANNModel
from gen_dataset_2012 import load_gt_regions, find_csv, label_bboxes

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
DANN_WEIGHT = "/home/hyw/chendawei/classification-weight/best_dann_model.pth"

FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -3.0
TILE_NORM_BLOCKSIZE = 0
MIN_SIZE = 15
DIST_UM = 5.0
PATCH = 224
BATCH = 256

NUM_DOMAIN_CLASSES = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def res_um(name):
    return 0.2456 if name.startswith("A") else (0.2273 + 0.22753) / 2.0


def load_gt_points(csv_p):
    pts = []
    for xs, ys in load_gt_regions(csv_p):
        pts.append((float(xs.mean()), float(ys.mean())))
    return pts


def match(dets, gts, thr_px):
    """dets: [(x,y,prob)], gts: [(x,y)]; 一对一贪心匹配。"""
    matched_g = set(); matched_d = set()
    if dets:
        dets_xy = np.array([(d[0], d[1]) for d in dets])
        for gi, (gx, gy) in enumerate(gts):
            dd = np.sqrt((dets_xy[:, 0] - gx) ** 2 + (dets_xy[:, 1] - gy) ** 2)
            order = np.argsort(dd)
            for di in order:
                if dd[di] > thr_px:
                    break
                if di in matched_d:
                    continue
                matched_d.add(int(di)); matched_g.add(gi); break
    tp = len(matched_g)
    fp = len(dets) - len(matched_d)
    fn = len(gts) - tp
    return tp, fp, fn


def crop_224(img_rgb, bbox):
    """照 224_patch_around_bbox.py：以 bbox 中心固定 224 窗口，越界向内平移。

    bbox = (left, top, right, bottom)，right/bottom 为开区间。
    """
    H, W = img_rgb.shape[:2]
    left, top, right, bottom = bbox
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    half = PATCH // 2
    l = max(0, cx - half); t = max(0, cy - half)
    r = l + PATCH; b = t + PATCH
    if r > W:
        r = W; l = r - PATCH
    if b > H:
        b = H; t = b - PATCH
    l = max(0, l); t = max(0, t)
    patch = img_rgb[t:b, l:r]
    if patch.shape[0] != PATCH or patch.shape[1] != PATCH:
        patch = cv2.resize(patch, (PATCH, PATCH), interpolation=cv2.INTER_CUBIC)
    return patch


def load_masks_from_cache(dst, base):
    mp = dst / "cellpose_cache" / "test_masks_cp3" / f"{base}_mask.npy"
    if not mp.exists():
        return None
    return np.load(mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=str(DST))
    ap.add_argument('--weight', default=DANN_WEIGHT)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--flow', type=float, default=FLOW_THRESHOLD)
    ap.add_argument('--cellprob', type=float, default=CELLPROB_THRESHOLD)
    ap.add_argument('--min-size', type=int, default=MIN_SIZE)
    ap.add_argument('--live', action='store_true', help='现场 Cellpose 推理(默认读缓存)')
    args = ap.parse_args()

    dst = Path(args.dst)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DANNModel(num_classes=2, num_domain_classes=NUM_DOMAIN_CLASSES, lambda_val=0.0)
    state = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)

    cp = None
    if args.live:
        from cellpose import models, core
        cp = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=CELLPOSE_WEIGHT)
    print(f"device={device}; masks={'live' if args.live else 'cache'}; weight={args.weight}", flush=True)

    img_dir = dst / "train-val-test" / "test-original-img"
    bmps = sorted(img_dir.glob("*.bmp"))

    records = []  # (base, cx, cy, prob)
    gt_by_hpf = {}
    total_gt = 0

    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img_bgr = cv2.imread(str(bmp))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        px = res_um(base)
        csv_p = find_csv(base)
        gts = load_gt_points(csv_p) if csv_p else []
        gt_by_hpf[base] = (gts, DIST_UM / px)
        total_gt += len(gts)

        if args.live:
            masks, _, _ = cp.eval(img_rgb, batch_size=1,
                                  flow_threshold=args.flow,
                                  cellprob_threshold=args.cellprob,
                                  min_size=args.min_size,
                                  normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE})
            mask = masks if isinstance(masks, np.ndarray) else np.array(masks)
        else:
            mask = load_masks_from_cache(dst, base)
            if mask is None:
                print(f"[{hi}/{len(bmps)}] {base}: 无缓存 mask，跳过", flush=True)
                continue

        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        bboxes = label_bboxes(mask)

        patches, items = [], []
        for lab, (cy, cx) in zip(labels, cents):
            y0, y1, x0, x1 = bboxes[lab]
            patch = crop_224(img_rgb, (x0, y0, x1, y1))
            patches.append(patch)
            items.append((float(cx), float(cy)))

        if patches:
            probs = []
            for i in range(0, len(patches), args.batch):
                chunk = patches[i:i + args.batch]
                tens = []
                for p in chunk:
                    x = p.astype(np.float32) / 255.0
                    x = (x - IMAGENET_MEAN) / IMAGENET_STD
                    tens.append(torch.from_numpy(x).permute(2, 0, 1).float())
                batch = torch.stack(tens, 0).to(device)
                with torch.no_grad():
                    logits = model.predict_only(batch)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p.tolist())
            for (cx, cy), p in zip(items, probs):
                records.append((base, cx, cy, float(p)))

        print(f"[{hi}/{len(bmps)}] {base}: masks={len(labels)} gt={len(gts)}", flush=True)

    print(f"\n总 GT: {total_gt}")
    print(f"{'阈值':>6} {'TP':>5} {'FP':>6} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    best = {'f1': -1}
    for thr in np.arange(0.05, 0.995, 0.01):
        by = {}
        for (base, cx, cy, p) in records:
            by.setdefault(base, []).append((cx, cy, p))
        tp = fp = fn = 0
        for base, recs in by.items():
            gts, thr_px = gt_by_hpf[base]
            dets = [(cx, cy, p) for (cx, cy, p) in recs if p >= thr]
            t, f, n = match(dets, gts, thr_px)
            tp += t; fp += f; fn += n
        pr = tp / (tp + fp) if (tp + fp) else 0
        rc = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
        if abs((thr * 100) % 5) < 1e-6:
            print(f"{thr:>6.2f} {tp:>5} {fp:>6} {fn:>4} {pr:>10.4f} {rc:>8.4f} {f1:>8.4f}")
        if f1 > best['f1']:
            best = {'f1': f1, 'thr': thr, 'tp': tp, 'fp': fp, 'fn': fn, 'p': pr, 'r': rc}
    print("=" * 60)
    print(f"最优阈值: {best['thr']:.2f}  F1={best['f1']:.4f}  P={best['p']:.4f}  R={best['r']:.4f}")
    print(f"TP={best['tp']}, FP={best['fp']}, FN={best['fn']}")


if __name__ == "__main__":
    main()
