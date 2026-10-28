#!/usr/bin/env python3
"""
ICPR2012 端到端评估（现场推理版，不读 test mask 缓存）：
  现场跑 Cellpose -> cell-on-white -> 分类 -> 5µm 质心 F1。

与 eval_2012.py 的差异：mask 由本脚本用 Cellpose 现场推理得到（仅内存，不落盘），
不读取 cellpose_cache/test_masks_cp3。

用法:
  python eval_2012_live.py --config <cfg> --checkpoint <pth>
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage
from mmpretrain import init_model
from mmengine.config import Config
from cellpose import models, core

from gen_dataset_2012 import cell_on_white, load_gt_regions, find_csv, label_bboxes

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -3.0
TILE_NORM_BLOCKSIZE = 0
MIN_SIZE = 15
DIST_UM = 5.0
BATCH = 64


def res_um(name):
    return 0.2456 if name.startswith("A") else (0.2273 + 0.22753) / 2.0


def load_gt_points(csv_p):
    """GT 区域质心 -> [(x, y)]"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dst', default=str(DST))
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--flow', type=float, default=FLOW_THRESHOLD)
    ap.add_argument('--cellprob', type=float, default=CELLPROB_THRESHOLD)
    ap.add_argument('--min-size', type=int, default=MIN_SIZE)
    args = ap.parse_args()

    dst = Path(args.dst)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(args.config)
    model = init_model(args.config, args.checkpoint, device=device)
    model.eval()
    mean = np.array(cfg.data_preprocessor['mean'], dtype=np.float32)
    std = np.array(cfg.data_preprocessor['std'], dtype=np.float32)
    to_rgb = cfg.data_preprocessor.get('to_rgb', True)

    # 若 val 管线含 MacenkoNorm，则对白板图先做同样归一化
    macenko = None
    for t in cfg.val_dataloader.dataset.pipeline:
        if isinstance(t, dict) and t.get('type') == 'MacenkoNorm':
            from my_transforms import MacenkoNorm
            macenko = MacenkoNorm(fit_file=t.get('fit_file'))
            print("应用 MacenkoNorm:", t.get('fit_file'))
            break

    # 现场 Cellpose 推理（不读/不写缓存）
    use_gpu = core.use_gpu()
    print(f"Cellpose GPU: {use_gpu}; checkpoint: {args.checkpoint}", flush=True)
    cp = models.CellposeModel(gpu=use_gpu, pretrained_model=CELLPOSE_WEIGHT)

    img_dir = dst / "train-val-test" / "test-original-img"

    records = []  # (hpf, cx, cy, prob)
    gt_by_hpf = {}
    total_gt = 0
    bmps = sorted(img_dir.glob("*.bmp"))
    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img = cv2.imread(str(bmp))            # BGR: 用于 cell_on_white（与训练一致）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB: 用于 Cellpose（与缓存/pipline 一致）
        px = res_um(base)
        csv_p = find_csv(base)
        gts = load_gt_points(csv_p) if csv_p else []
        gt_by_hpf[base] = (gts, DIST_UM / px)
        total_gt += len(gts)

        masks, _, _ = cp.eval(img_rgb, batch_size=1,
                              flow_threshold=args.flow,
                              cellprob_threshold=args.cellprob,
                              min_size=args.min_size,
                              normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE})
        mask = masks if isinstance(masks, np.ndarray) else np.array(masks)

        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        bboxes = label_bboxes(mask)
        items = []
        boards = []
        for lab, (cy, cx) in zip(labels, cents):
            b = cell_on_white(img, mask, lab, bbox=bboxes.get(lab))
            if b is not None:
                if macenko is not None:
                    b = macenko.transform({'img': b})['img']
                boards.append(b)
                items.append((lab, cy, cx))
        if boards:
            probs = []
            for i in range(0, len(boards), args.batch):
                chunk = boards[i:i + args.batch]
                tens = []
                for b in chunk:
                    x = cv2.cvtColor(b, cv2.COLOR_BGR2RGB) if to_rgb else b
                    x = x.astype(np.float32)
                    x = (x - mean) / std
                    tens.append(torch.from_numpy(x).permute(2, 0, 1).float())
                batch = torch.stack(tens, 0).to(device)
                with torch.no_grad():
                    logits = model(batch, mode='tensor')
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(p.tolist())
            for (lab, cy, cx), p in zip(items, probs):
                records.append((base, float(cx), float(cy), float(p)))
        print(f"[{hi}/{len(bmps)}] {base}: masks={len(labels)} gt={len(gts)}", flush=True)

    # 阈值扫描
    print(f"\n总 GT: {total_gt}")
    print(f"{'阈值':>6} {'TP':>5} {'FP':>6} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    best = {'f1': -1}
    for thr in np.arange(0.05, 0.96, 0.01):
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
