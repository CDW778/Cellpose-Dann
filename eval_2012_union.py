#!/usr/bin/env python3
"""DANN ∪ ConvNeXt 并集评估（检测层面 OR，不做分数平均）。

两分支对同一批 Cellpose mask 打分：
  p_D: DANN, 224 原始 RGB 窗口 + ImageNet 归一化
  p_C: ConvNeXt, cell-on-white 白板 + 旧 config 预处理(0-255)
检测集：D_d={p_D>=t_D}, D_c={p_C>=t_C}, 并集 D_u=D_d∪D_c（同 mask 去重）
对 D_u 重新做 5µm 一对一匹配 -> TP/FP/FN/F1。
阈值：各分支在 val(03) 上各自取最优；test(04) 只评估。
"""
import argparse
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage

from mmpretrain import init_model
from mmengine.config import Config

from dann_model import DANNModel
from gen_dataset_2012 import cell_on_white, find_csv, label_bboxes
from eval_2012_dann_live import res_um, load_gt_points, match

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OLD = "/home/hyw/chendawei/cellpose-4.2.1.1-opencode/mmpretrain-main"
CV_CONFIG = OLD + "/configs/convnext_v2/my1-convnext-v2-tiny_32xb32_in1k.py"
CV_CKPT = OLD + "/tools/work_dirs/my1-convnext-v2-tiny_2012/best_single-label_f1-score_epoch_174.pth"
DANN_WEIGHT = "/home/hyw/chendawei/dann_finetune/best_s3_seed0.pth"
NUM_DOMAIN_CLASSES = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 224
CHUNK = 256
CACHE_DIR = Path("/home/hyw/chendawei/dann_finetune")


def crop_224(img_rgb, bbox):
    H, W = img_rgb.shape[:2]
    left, top, right, bottom = bbox
    cx = int((left + right) / 2); cy = int((top + bottom) / 2)
    half = PATCH // 2
    l = max(0, cx - half); t = max(0, cy - half)
    r = l + PATCH; b = t + PATCH
    if r > W: r = W; l = r - PATCH
    if b > H: b = H; t = b - PATCH
    l = max(0, l); t = max(0, t)
    p = img_rgb[t:b, l:r]
    if p.shape[0] != PATCH or p.shape[1] != PATCH:
        p = cv2.resize(p, (PATCH, PATCH), interpolation=cv2.INTER_CUBIC)
    return p


def gt_for_split(split):
    gt = {}
    for bmp in sorted((DST / "train-val-test" / f"{split}-original-img").glob("*.bmp")):
        base = bmp.stem
        csv_p = find_csv(base)
        pts = load_gt_points(csv_p) if csv_p else []
        gt[base] = (pts, 5.0 / res_um(base))
    return gt


def score_split(split, dann, conv, cv_mean, cv_std, cv_to_rgb, device, use_cache=True):
    cache = CACHE_DIR / f"union_scores_{split}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        print(f"[{split}] 读取缓存 {cache}", flush=True)
        return d
    img_dir = DST / "train-val-test" / f"{split}-original-img"
    mask_dir = DST / "cellpose_cache" / f"{split}_masks_cp3"
    rows = []
    bmps = sorted(img_dir.glob("*.bmp"))
    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img_bgr = cv2.imread(str(bmp))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_dir / f"{base}_mask.npy")
        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        bboxes = label_bboxes(mask)
        coord = {int(l): (float(c[1]), float(c[0])) for l, c in zip(labels, cents)}
        for i in range(0, len(labels), CHUNK):
            chunk = labels[i:i + CHUNK]
            raw, board = [], []
            for lab in chunk:
                y0, y1, x0, x1 = bboxes[lab]
                raw.append(crop_224(img_rgb, (x0, y0, x1, y1)))
                b = cell_on_white(img_bgr, mask, lab, bbox=bboxes.get(lab))
                board.append(b)
            # DANN
            tens = []
            for p in raw:
                x = p.astype(np.float32) / 255.0
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
                tens.append(torch.from_numpy(x).permute(2, 0, 1).float())
            with torch.no_grad():
                pD = torch.softmax(dann.predict_only(torch.stack(tens).to(device)), 1)[:, 1].cpu().numpy()
            # ConvNeXt
            valid = [k for k, b in enumerate(board) if b is not None]
            pC = np.zeros(len(chunk), np.float32)
            if valid:
                ct = []
                for k in valid:
                    x = cv2.cvtColor(board[k], cv2.COLOR_BGR2RGB) if cv_to_rgb else board[k]
                    x = (x.astype(np.float32) - cv_mean) / cv_std
                    ct.append(torch.from_numpy(x).permute(2, 0, 1).float())
                with torch.no_grad():
                    lg = conv(torch.stack(ct).to(device), mode='tensor')
                    pv = torch.softmax(lg, 1)[:, 1].cpu().numpy()
                for k, v in zip(valid, pv):
                    pC[k] = float(v)
            for k, lab in enumerate(chunk):
                cx, cy = coord[lab]
                rows.append((base, lab, cx, cy, float(pD[k]), float(pC[k])))
        print(f"[{split}] {hi}/{len(bmps)} {base}: masks={len(labels)}", flush=True)
    base_a = np.array([r[0] for r in rows])
    lab_a = np.array([r[1] for r in rows], np.int64)
    cx_a = np.array([r[2] for r in rows], np.float32)
    cy_a = np.array([r[3] for r in rows], np.float32)
    pd_a = np.array([r[4] for r in rows], np.float32)
    pc_a = np.array([r[5] for r in rows], np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, base=base_a, label=lab_a, cx=cx_a, cy=cy_a, pD=pd_a, pC=pc_a)
    print(f"[{split}] 写出缓存 {cache} n={len(rows)}", flush=True)
    return dict(base=base_a, label=lab_a, cx=cx_a, cy=cy_a, pD=pd_a, pC=pc_a)


def records(data, split):
    gt = gt_for_split(split)
    out = []
    for i in range(len(data['base'])):
        b = str(data['base'][i])
        out.append(dict(base=b, label=int(data['label'][i]), cx=float(data['cx'][i]),
                        cy=float(data['cy'][i]), pD=float(data['pD'][i]), pC=float(data['pC'][i]),
                        gt=gt[b]))
    return out


def eval_by(by):
    tp = fp = fn = 0
    for base, dets in by.items():
        if not dets:
            continue
        pts, thr_px = dets[0][3], dets[0][4]
        t, f, n = match([(d[0], d[1], d[2]) for d in dets], pts, thr_px)
        tp += t; fp += f; fn += n
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def group(records_, key, thr, R_um=None):
    by = {}
    for r in records_:
        if r[key] >= thr:
            by.setdefault(r['base'], []).append((r['cx'], r['cy'], r[key], r['gt'][0], r['gt'][1]))
    if R_um is not None:
        for base in by:
            px = res_um(base)
            rpx = R_um / px
            kept = []
            for d in sorted(by[base], key=lambda t: -t[2]):
                if all(np.hypot(d[0] - k[0], d[1] - k[1]) > rpx for k in kept):
                    kept.append(d)
            by[base] = kept
    return by


def group_union(records_, tD, tC, R_um=None):
    by = {}
    for r in records_:
        if r['pD'] >= tD or r['pC'] >= tC:
            sc = max(r['pD'], r['pC'])
            by.setdefault(r['base'], []).append((r['cx'], r['cy'], sc, r['gt'][0], r['gt'][1]))
    if R_um is not None:
        for base in by:
            rpx = R_um / res_um(base)
            kept = []
            for d in sorted(by[base], key=lambda t: -t[2]):
                if all(np.hypot(d[0] - k[0], d[1] - k[1]) > rpx for k in kept):
                    kept.append(d)
            by[base] = kept
    return by


def scan_thr(records_, key):
    best = {'f1': -1}
    for thr in np.arange(0.05, 0.995, 0.01):
        tp, fp, fn = eval_by(group(records_, key, thr))
        p, r, f = prf(tp, fp, fn)
        if f > best['f1']:
            best = dict(f1=f, thr=thr, tp=tp, fp=fp, fn=fn, p=p, r=r)
    return best


def evaluate(records_, tD, tC, R_um=None):
    out = {}
    for name, by in [('DANN', group(records_, 'pD', tD)),
                     ('Conv', group(records_, 'pC', tC)),
                     ('Union', group_union(records_, tD, tC)),
                     ('Union+NMS', group_union(records_, tD, tC, R_um))]:
        tp, fp, fn = eval_by(by)
        p, r, f = prf(tp, fp, fn)
        out[name] = dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f)
    return out


def complementarity(records_, tD, tC):
    # per GT: has_D / has_C within 5um (GT-centric coverage)
    byD = group(records_, 'pD', tD)
    byC = group(records_, 'pC', tC)
    a = b = c = d = 0
    gt = {}
    for r in records_:
        gt[r['base']] = r['gt']
    for base, (pts, thr_px) in gt.items():
        dD = [(x, y) for (x, y, *_rest) in byD.get(base, [])]
        dC = [(x, y) for (x, y, *_rest) in byC.get(base, [])]
        for gx, gy in pts:
            hD = any(np.hypot(x - gx, y - gy) <= thr_px for x, y in dD)
            hC = any(np.hypot(x - gx, y - gy) <= thr_px for x, y in dC)
            if hD and hC: a += 1
            elif hD and not hC: b += 1
            elif (not hD) and hC: c += 1
            else: d += 1
    return a, b, c, d


def fmt(name, m):
    return (f"{name:10s} TP={m['tp']:3d} FP={m['fp']:3d} FN={m['fn']:3d} "
            f"P={m['p']:.4f} R={m['r']:.4f} F1={m['f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-cache', action='store_true', default=True)
    ap.add_argument('--no-cache', dest='use_cache', action='store_false')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.fromfile(CV_CONFIG)
    cv_mean = np.array(cfg.data_preprocessor['mean'], np.float32)
    cv_std = np.array(cfg.data_preprocessor['std'], np.float32)
    cv_to_rgb = cfg.data_preprocessor.get('to_rgb', True)
    conv = init_model(CV_CONFIG, CV_CKPT, device=device); conv.eval()
    dann = DANNModel(2, NUM_DOMAIN_CLASSES, 0.0)
    dann.load_state_dict(torch.load(DANN_WEIGHT, map_location='cpu')); dann.eval().to(device)
    print(f"loaded ConvNeXt + DANN on {device}", flush=True)

    val = records(score_split('val', dann, conv, cv_mean, cv_std, cv_to_rgb, device, args.use_cache), 'val')
    test = records(score_split('test', dann, conv, cv_mean, cv_std, cv_to_rgb, device, args.use_cache), 'test')

    bD = scan_thr(val, 'pD'); bC = scan_thr(val, 'pC')
    tD, tC = bD['thr'], bC['thr']
    print(f"\n== val 选阈值 ==\nDANN: thr={tD:.2f} F1={bD['f1']:.4f} (P={bD['p']:.3f} R={bD['r']:.3f})")
    print(f"Conv: thr={tC:.2f} F1={bC['f1']:.4f} (P={bC['p']:.3f} R={bC['r']:.3f})")

    # NMS 半径在 val 上选
    bestR = None; bestF = -1
    for R in [5.0, 7.5, 10.0]:
        byu = group_union(val, tD, tC, R)
        tp, fp, fn = eval_by(byu); _, _, f = prf(tp, fp, fn)
        print(f"val Union+NMS R={R}µm F1={f:.4f}")
        if f > bestF: bestF = f; bestR = R
    print(f"选定 NMS 半径 R={bestR}µm (val)")

    res = evaluate(test, tD, tC, bestR)
    print("\n===== test(04) e2e 5µm F1（阈值来自 val）=====")
    for k in ['DANN', 'Conv', 'Union', 'Union+NMS']:
        print(fmt(k, res[k]))

    # test-oracle 参考（仅参考，不用于选阈值）
    oD = scan_thr(test, 'pD'); oC = scan_thr(test, 'pC')
    print("\n[参考] test-oracle 单分支:")
    print(f"  DANN thr={oD['thr']:.2f} F1={oD['f1']:.4f} P={oD['p']:.4f} R={oD['r']:.4f} TP={oD['tp']} FP={oD['fp']} FN={oD['fn']}")
    print(f"  Conv thr={oC['thr']:.2f} F1={oC['f1']:.4f} P={oC['p']:.4f} R={oC['r']:.4f} TP={oC['tp']} FP={oC['fp']} FN={oC['fn']}")

    a, b, c, d = complementarity(test, tD, tC)
    print("\n===== 互补性（test，GT 5µm 覆盖；基于 val 阈值）=====")
    print("                Conv命中   Conv漏")
    print(f"  DANN命中      {a:6d}   {b:6d}")
    print(f"  DANN漏        {c:6d}   {d:6d}")
    print(f"  => DANN 漏但 Conv 补回: {c} / 84; 两者都漏: {d} / 84")


if __name__ == '__main__':
    main()
