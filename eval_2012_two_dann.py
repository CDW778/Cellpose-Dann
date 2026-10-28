#!/usr/bin/env python3
"""224 与 112 两个 DANN 的并行评估：val 选阈值/融合，test 只评估。

融合：average / max / selective-union（112 检测仅在离 224 检测 >R 时加入）。
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage

from dann_model import DANNModel
from dann_dataset import crop_patch
from gen_dataset_2012 import find_csv, load_gt_regions, label_bboxes

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT = Path("/home/hyw/chendawei/dann_finetune")
NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CHUNK = 128


def res_um(b):
    return 0.2456 if b.startswith("A") else (0.2273 + 0.22753) / 2.0


def gt_map(split):
    out = {}
    for bmp in sorted((DST / "train-val-test" / f"{split}-original-img").glob("*.bmp")):
        base = bmp.stem
        p = find_csv(base)
        regs = load_gt_regions(p) if p else []
        out[base] = ([(float(xs.mean()), float(ys.mean())) for xs, ys in regs], 5.0 / res_um(base))
    return out


def match(dets, gts, thr):
    mg = set(); md = set()
    if dets:
        xy = np.array([(d[0], d[1]) for d in dets])
        for gi, (gx, gy) in enumerate(gts):
            dd = np.sqrt((xy[:, 0] - gx) ** 2 + (xy[:, 1] - gy) ** 2)
            for di in np.argsort(dd):
                if dd[di] > thr: break
                if di in md: continue
                md.add(int(di)); mg.add(gi); break
    return len(mg), len(dets) - len(md), len(gts) - len(mg)


def norm_stack(patches):
    ts = []
    for p in patches:
        x = p.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        ts.append(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float())
    return torch.stack(ts, 0)


def score_split(split, m224, m112, device, use_cache=True):
    cache = OUT / f"two_dann_scores_{split}.npz"
    if use_cache and cache.exists():
        print(f"[{split}] 读取缓存 {cache}", flush=True)
        return np.load(cache, allow_pickle=True)
    img_dir = DST / "train-val-test" / f"{split}-original-img"
    mask_dir = DST / "cellpose_cache" / f"{split}_masks_cp3"
    rows = []
    bmps = sorted(img_dir.glob("*.bmp"))
    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_dir / f"{base}_mask.npy")
        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        bb = label_bboxes(mask)
        coord = {int(l): (float(c[1]), float(c[0])) for l, c in zip(labels, cents)}
        for i in range(0, len(labels), CHUNK):
            chunk = labels[i:i + CHUNK]
            p224 = [crop_patch(img, (bb[l][2], bb[l][0], bb[l][3], bb[l][1]), 224) for l in chunk]
            p112 = [crop_patch(img, (bb[l][2], bb[l][0], bb[l][3], bb[l][1]), 112) for l in chunk]
            with torch.no_grad():
                a = torch.softmax(m224.predict_only(norm_stack(p224).to(device)), 1)[:, 1].cpu().numpy()
                b = torch.softmax(m112.predict_only(norm_stack(p112).to(device)), 1)[:, 1].cpu().numpy()
            for k, l in enumerate(chunk):
                cx, cy = coord[l]
                rows.append((base, l, cx, cy, float(a[k]), float(b[k])))
        print(f"[{split}] {hi}/{len(bmps)} {base}: masks={len(labels)}", flush=True)
    base_a = np.array([r[0] for r in rows])
    lab_a = np.array([r[1] for r in rows], np.int64)
    cx_a = np.array([r[2] for r in rows], np.float32)
    cy_a = np.array([r[3] for r in rows], np.float32)
    p224_a = np.array([r[4] for r in rows], np.float32)
    p112_a = np.array([r[5] for r in rows], np.float32)
    np.savez_compressed(cache, base=base_a, label=lab_a, cx=cx_a, cy=cy_a, p224=p224_a, p112=p112_a)
    print(f"[{split}] 写出缓存 {cache} n={len(rows)}", flush=True)
    return np.load(cache, allow_pickle=True)


def records_from(d, split):
    gt = gt_map(split)
    recs = []
    for i in range(len(d['base'])):
        b = str(d['base'][i])
        p224 = float(d['p224'][i]); p112 = float(d['p112'][i])
        recs.append(dict(base=b, cx=float(d['cx'][i]), cy=float(d['cy'][i]),
                         p224=p224, p112=p112, avg=(p224 + p112) / 2, mx=max(p224, p112),
                         gts=gt[b][0], thr=gt[b][1]))
    return recs


def eval_key(recs, key, thr):
    by = {}
    for r in recs:
        if r[key] >= thr:
            by.setdefault(r['base'], []).append((r['cx'], r['cy'], r[key], r['gts'], r['thr']))
    tp = fp = fn = 0
    for base, ds in by.items():
        t, f, n = match([(x, y, s) for x, y, s, _, _ in ds], ds[0][3], ds[0][4])
        tp += t; fp += f; fn += n
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return tp, fp, fn, p, r, f1


def scan(recs, key):
    best = {'f1': -1}
    for thr in np.arange(0.05, 0.995, 0.01):
        tp, fp, fn, p, r, f1 = eval_key(recs, key, thr)
        if f1 > best['f1']:
            best = dict(f1=f1, thr=thr, tp=tp, fp=fp, fn=fn, p=p, r=r)
    return best


def eval_selective(recs, t224, t112, R_um):
    by = {}
    for r in recs:
        by.setdefault(r['base'], []).append(r)
    tp = fp = fn = 0
    for base, rs in by.items():
        gts = rs[0]['gts']; thr = rs[0]['thr']
        dets = [(r['cx'], r['cy'], r['p224']) for r in rs if r['p224'] >= t224]
        coords = [(d[0], d[1]) for d in dets]
        rpx = R_um / res_um(base)
        for r in rs:
            if r['p112'] >= t112 and r['p224'] < t224:
                if all(np.hypot(r['cx'] - x, r['cy'] - y) > rpx for x, y in coords):
                    dets.append((r['cx'], r['cy'], r['p112']))
        t, f, n = match(dets, gts, thr)
        tp += t; fp += f; fn += n
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return tp, fp, fn, p, r, f1


def show(name, m):
    tp, fp, fn, p, r, f1 = m
    print(f"{name:22s} TP={tp:3d} FP={fp:3d} FN={fn:3d} P={p:.4f} R={r:.4f} F1={f1:.4f}", flush=True)


def complementarity(recs, t224, t112):
    by = {}
    for r in recs:
        by.setdefault(r['base'], []).append(r)
    a = b = c = d = 0
    for base, rs in by.items():
        gts = rs[0]['gts']; thr = rs[0]['thr']
        for gx, gy in gts:
            h224 = any(r['p224'] >= t224 and np.hypot(r['cx'] - gx, r['cy'] - gy) <= thr for r in rs)
            h112 = any(r['p112'] >= t112 and np.hypot(r['cx'] - gx, r['cy'] - gy) <= thr for r in rs)
            if h224 and h112: a += 1
            elif h224: b += 1
            elif h112: c += 1
            else: d += 1
    return a, b, c, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w224', default=str(OUT / "best_s3_seed0.pth"))
    ap.add_argument('--w112', default=str(OUT / "best_s3_112_seed0.pth"))
    ap.add_argument('--use-cache', action='store_true', default=True)
    ap.add_argument('--no-cache', dest='use_cache', action='store_false')
    ap.add_argument('--selective-R', type=float, default=10.0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m224 = DANNModel(2, NDC, 0.0); m224.load_state_dict(torch.load(args.w224, map_location='cpu')); m224.eval().to(device)
    m112 = DANNModel(2, NDC, 0.0); m112.load_state_dict(torch.load(args.w112, map_location='cpu')); m112.eval().to(device)
    print(f"loaded 224={args.w224}\n       112={args.w112}", flush=True)

    dv = score_split('val', m224, m112, device, args.use_cache)
    dt = score_split('test', m224, m112, device, args.use_cache)
    val = records_from(dv, 'val'); test = records_from(dt, 'test')

    b224 = scan(val, 'p224'); b112 = scan(val, 'p112')
    bavg = scan(val, 'avg'); bmax = scan(val, 'mx')
    sel_val = eval_selective(val, b224['thr'], b112['thr'], args.selective_R)
    print("\n== val 选择 ==")
    print(f"224 thr={b224['thr']:.2f} F1={b224['f1']:.4f} | 112 thr={b112['thr']:.2f} F1={b112['f1']:.4f}")
    print(f"avg thr={bavg['thr']:.2f} F1={bavg['f1']:.4f} | max thr={bmax['thr']:.2f} F1={bmax['f1']:.4f}")
    print(f"selective(224thr,112thr,R={args.selective_R}) F1={sel_val[5]:.4f}")

    methods = {
        '224-only': eval_key(test, 'p224', b224['thr']),
        '112-only': eval_key(test, 'p112', b112['thr']),
        'avg': eval_key(test, 'avg', bavg['thr']),
        'max': eval_key(test, 'mx', bmax['thr']),
        'selective': eval_selective(test, b224['thr'], b112['thr'], args.selective_R),
    }
    print("\n===== test(04) e2e 5µm（阈值/规则来自 val）=====")
    order = ['224-only', '112-only', 'avg', 'max', 'selective']
    for k in order:
        show(k, methods[k])

    val_f1 = {'224-only': b224['f1'], '112-only': b112['f1'], 'avg': bavg['f1'],
              'max': bmax['f1'], 'selective': sel_val[5]}
    chosen = max(val_f1, key=val_f1.get)
    print(f"\nval 最优方法: {chosen} (val F1={val_f1[chosen]:.4f}) -> test F1={methods[chosen][5]:.4f}")

    a, b, c, d = complementarity(test, b224['thr'], b112['thr'])
    print("\n===== 互补性（test, val 阈值）=====")
    print("              112命中  112漏")
    print(f"  224命中     {a:6d}  {b:6d}")
    print(f"  224漏       {c:6d}  {d:6d}")
    print(f"  => 224 漏但 112 补回: {c} / {a+b+c+d}")


if __name__ == '__main__':
    main()
