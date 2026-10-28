#!/usr/bin/env python3
"""ICPR2014 官方 test 评估：Cellpose(cpsam_v2) + DANN(224) + 8µm 匹配。

- Aperio: 0.2455 µm/px（各向同性）
- Hamamatsu: 0.227299 (x) / 0.227531 (y) µm/px（各向异性）
- GT: {slide}/mitosis/{frame}_mitosis.csv，每行 x,y,confidence
- 判定：预测中心与 GT 中心距离 <= 8µm
- 主结果：固定阈值 0.93（2012 val 选出）；另报 2014 test oracle 参考
"""
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage

from cellpose import models, core
from dann_model import DANNModel
from dann_dataset import crop_patch
from gen_dataset_2012 import label_bboxes

ROOT = Path("/home/hyw/dataset/chendawei/ICPR2014")
OUT = Path("/home/hyw/chendawei/icpr2014_eval")
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
DANN_WEIGHT = "/home/hyw/chendawei/dann_finetune/best_s3_seed0.pth"
NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FLOW = 1.0
CELLPROB = -3.0
MIN_SIZE = 15
TILE_NORM = 0
TILE_OVERLAP = 0.1
BATCH = 16
PATCH = 224
DANN_BATCH = 256
DIST_UM = 8.0
FIXED_THR = 0.93

SCANNERS = {
    "aperio": dict(slides=["A06", "A08", "A09", "A13", "A16"], res=(0.2455, 0.2455)),
    "hamamatsu": dict(slides=["H06", "H08", "H09", "H13", "H16"], res=(0.227299, 0.227531)),
}


def dist_um(dx, dy, res):
    return float(np.sqrt((dx * res[0]) ** 2 + (dy * res[1]) ** 2))


def load_gt(csv_p):
    pts = []
    if not csv_p.exists():
        return pts
    for line in open(csv_p):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(";", ",").split(",")
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except Exception:
            continue
    return pts


def match(dets, gts, res):
    used = set(); matched = 0
    for gx, gy in gts:
        best = None
        for di, (x, y, p) in enumerate(dets):
            if di in used:
                continue
            d = dist_um(x - gx, y - gy, res)
            if d <= DIST_UM and (best is None or d < best[0]):
                best = (d, di)
        if best is not None:
            used.add(best[1]); matched += 1
    return matched, len(dets) - matched, len(gts) - matched


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def get_mask(cp, frame, tiff, scanner):
    mp = OUT / "masks" / scanner / f"{frame}_mask.npy"
    if mp.exists():
        return np.load(mp)
    img = cv2.cvtColor(cv2.imread(str(tiff)), cv2.COLOR_BGR2RGB)
    masks, _, _ = cp.eval(img, batch_size=BATCH, flow_threshold=FLOW,
                          cellprob_threshold=CELLPROB, min_size=MIN_SIZE,
                          normalize={"tile_norm_blocksize": TILE_NORM},
                          tile_overlap=TILE_OVERLAP, augment=False)
    m = masks if isinstance(masks, np.ndarray) else np.array(masks)
    mp.parent.mkdir(parents=True, exist_ok=True)
    np.save(mp, m)
    return m


def score_frame(dann, frame, tiff, mask, scanner, device):
    sp = OUT / "scores" / scanner / f"{frame}.npz"
    if sp.exists():
        d = np.load(sp, allow_pickle=True)
        return d['cx'], d['cy'], d['p'], d['label']
    img = cv2.cvtColor(cv2.imread(str(tiff)), cv2.COLOR_BGR2RGB)
    labels = [int(l) for l in np.unique(mask) if l != 0]
    cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
    bb = label_bboxes(mask)
    cxs, cys, ps = [], [], []
    for i in range(0, len(labels), DANN_BATCH):
        chunk = labels[i:i + DANN_BATCH]
        patch = [crop_patch(img, (bb[l][2], bb[l][0], bb[l][3], bb[l][1]), PATCH) for l in chunk]
        tens = []
        for p in patch:
            x = p.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            tens.append(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float())
        with torch.no_grad():
            prob = torch.softmax(dann.predict_only(torch.stack(tens).to(device)), 1)[:, 1].cpu().numpy()
        for k in range(len(chunk)):
            cxs.append(float(cents[i + k][1])); cys.append(float(cents[i + k][0])); ps.append(float(prob[k]))
    sp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sp, cx=np.array(cxs), cy=np.array(cys), p=np.array(ps), label=np.array(labels))
    return np.array(cxs), np.array(cys), np.array(ps), np.array(labels)


def eval_scanner(scanner, cp, dann, device, max_frames=0):
    cfg = SCANNERS[scanner]
    frames = []
    for slide in cfg['slides']:
        for tiff in sorted((ROOT / f"mitos_atypia_2014_testing_{scanner}" / slide / "frames" / "x40").glob("*.tiff")):
            frames.append((slide, tiff))
    if max_frames:
        frames = frames[:max_frames]
    print(f"\n[{scanner}] frames={len(frames)}", flush=True)

    recs_by_frame = {}; gt_by = {}
    for i, (slide, tiff) in enumerate(frames, 1):
        frame = tiff.stem
        t0 = time.time()
        mask = get_mask(cp, frame, tiff, scanner)
        cx, cy, p, lab = score_frame(dann, frame, tiff, mask, scanner, device)
        gt = load_gt(ROOT / f"mitos_atypia_2014_testing_{scanner}" / slide / "mitosis" / f"{frame}_mitosis.csv")
        gt_by[frame] = gt
        recs_by_frame[frame] = dict(x=cx, y=cy, p=p)
        print(f"[{scanner}] {i}/{len(frames)} {frame}: masks={len(cx)} gt={len(gt)} {time.time()-t0:.1f}s", flush=True)
    return recs_by_frame, gt_by, cfg['res']


def eval_at(pairs, thr):
    tp = fp = fn = 0
    for recs_by_frame, gt_by, res in pairs:
        for frame, gt in gt_by.items():
            if frame not in recs_by_frame:
                continue
            d = recs_by_frame[frame]
            m = d['p'] >= thr
            dets = [(float(x), float(y), float(q)) for x, y, q in zip(d['x'][m], d['y'][m], d['p'][m])]
            t, f, n = match(dets, gt, res); tp += t; fp += f; fn += n
    return tp, fp, fn


def report(name, pairs):
    tp, fp, fn = eval_at(pairs, FIXED_THR)
    p, r, f1 = prf(tp, fp, fn)
    print(f"{name} FIXED thr={FIXED_THR}: TP={tp} FP={fp} FN={fn} P={p:.4f} R={r:.4f} F1={f1:.4f}", flush=True)
    best = {'f1': -1}
    for thr in np.arange(0.05, 0.995, 0.01):
        t, f, n = eval_at(pairs, thr)
        pp, rr, ff = prf(t, f, n)
        if ff > best['f1']:
            best = dict(f1=ff, thr=thr, tp=t, fp=f, fn=n, p=pp, r=rr)
    print(f"{name} ORACLE thr={best['thr']:.2f}: TP={best['tp']} FP={best['fp']} FN={best['fn']} "
          f"P={best['p']:.4f} R={best['r']:.4f} F1={best['f1']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-frames', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["masks/aperio", "masks/hamamatsu", "scores/aperio", "scores/hamamatsu"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}; loading models ...", flush=True)
    cp = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=CELLPOSE_WEIGHT)
    dann = DANNModel(2, NDC, 0.0)
    dann.load_state_dict(torch.load(DANN_WEIGHT, map_location='cpu'))
    dann.eval().to(device)
    print("models loaded", flush=True)

    pairs = {}
    for scanner in ["aperio", "hamamatsu"]:
        recs, gt_by, res = eval_scanner(scanner, cp, dann, device, args.max_frames)
        pairs[scanner] = (recs, gt_by, res)
        print("", flush=True)
        report(scanner.upper(), [pairs[scanner]])
    print("\n===== COMBINED =====", flush=True)
    report("COMBINED", [pairs["aperio"], pairs["hamamatsu"]])


if __name__ == '__main__':
    main()
