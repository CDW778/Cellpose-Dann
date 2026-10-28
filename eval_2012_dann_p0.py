#!/usr/bin/env python3
"""ICPR2012 评估 P0 版 DANN（GeM/CBAM/OD gate + 可选 cellprob 第4通道）。"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import ndimage

from dann_model import DANNModel
from dann_dataset import crop_patch, crop_map, crop_at, soft_mask_blend
from gen_dataset_2012 import find_csv, label_bboxes, load_gt_regions

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def res_um(b):
    return 0.2456 if b.startswith("A") else (0.2273 + 0.22753) / 2.0


def gt_points(base):
    p = find_csv(base)
    regs = load_gt_regions(p) if p else []
    return [(float(xs.mean()), float(ys.mean())) for xs, ys in regs], 5.0 / res_um(base)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--in-chans', type=int, default=3)
    ap.add_argument('--cellprob-dir', default=str(DST / 'cellprob_cache'))
    ap.add_argument('--mask-dir', default=None)
    ap.add_argument('--crop-center', choices=['bbox', 'centroid'], default='bbox')
    ap.add_argument('--soft-mask', action='store_true')
    ap.add_argument('--mask-channel', action='store_true')
    ap.add_argument('--mask-sigma', type=float, default=4.0)
    ap.add_argument('--mask-dilate', type=int, default=4)
    ap.add_argument('--mask-bg-sigma', type=float, default=20.0)
    ap.add_argument('--use-gem', action='store_true')
    ap.add_argument('--use-cbam', action='store_true')
    ap.add_argument('--use-od-gate', action='store_true')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--fixed-thr', type=float, default=None)
    ap.add_argument('--dump-scores', default=None, help='保存每个候选的 base/label/cx/cy/p')
    ap.add_argument('--tta', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DANNModel(2, NDC, 0.0, in_chans=args.in_chans, use_gem=args.use_gem,
                      use_cbam=args.use_cbam, use_od_gate=args.use_od_gate)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval().to(device)
    print(f"device={device}; weight={args.weight}; in_chans={args.in_chans}", flush=True)

    img_dir = DST / "train-val-test" / f"{args.split}-original-img"
    mask_dir = Path(args.mask_dir) if args.mask_dir else DST / "cellpose_cache" / f"{args.split}_masks_cp3"
    cp_dir = Path(args.cellprob_dir) / args.split
    if args.soft_mask and args.in_chans != 3:
        raise ValueError('soft_mask 目前只支持 in_chans=3')
    if args.mask_channel and args.in_chans != 4:
        raise ValueError('mask_channel 需要 in_chans=4')
    if args.soft_mask and args.mask_channel:
        raise ValueError('soft_mask 与 mask_channel 不能同时开启')
    print(f"crop_center={args.crop_center} soft_mask={args.soft_mask} "
          f"mask_sigma={args.mask_sigma} mask_dilate={args.mask_dilate} "
          f"mask_bg_sigma={args.mask_bg_sigma}", flush=True)
    recs = []; rec_labels = []; gt_by = {}; total_gt = 0
    bmps = sorted(img_dir.glob("*.bmp"))
    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_dir / f"{base}_mask.npy")
        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        cent_map = {int(lab): (float(c[0]), float(c[1])) for lab, c in zip(labels, cents)}
        bb = label_bboxes(mask)
        gts, thr_px = gt_points(base)
        gt_by[base] = (gts, thr_px); total_gt += len(gts)
        cp_map = np.load(cp_dir / f"{base}_cellprob.npy") if args.in_chans == 4 else None
        patches = []; coords = []
        for i in range(0, len(labels), args.batch):
            chunk = labels[i:i + args.batch]
            batch = []
            for lab in chunk:
                bbox = (bb[lab][2], bb[lab][0], bb[lab][3], bb[lab][1])
                win = None
                if args.soft_mask or args.mask_channel or args.crop_center == 'centroid':
                    if args.crop_center == 'centroid' and lab in cent_map:
                        cy_, cx_ = cent_map[lab]
                    else:
                        cx_ = (bbox[0] + bbox[2]) / 2.0
                        cy_ = (bbox[1] + bbox[3]) / 2.0
                    p, (l, t, r, b) = crop_at(img, cx_, cy_, args.input_size, cv2.INTER_CUBIC)
                    if args.soft_mask or args.mask_channel:
                        mask_c = mask[t:b, l:r]
                        comp_c = (mask_c == lab).astype(np.uint8)
                        if comp_c.shape[0] != args.input_size or comp_c.shape[1] != args.input_size:
                            comp_c = cv2.resize(comp_c, (args.input_size, args.input_size),
                                                interpolation=cv2.INTER_NEAREST)
                    if args.soft_mask:
                        p = soft_mask_blend(p, comp_c, args.mask_sigma, args.mask_dilate,
                                            args.mask_bg_sigma)
                    win = (l, t, r, b)
                else:
                    p = crop_patch(img, bbox, args.input_size)
                if args.in_chans == 4:
                    if args.mask_channel:
                        p = np.concatenate([p, comp_c[..., None]], axis=2)
                    else:
                        if win is None:
                            cp = crop_map(cp_map, bbox, args.input_size)
                        else:
                            l, t, r, b = win
                            cp = cp_map[t:b, l:r].astype(np.float32)
                            if cp.shape[0] != args.input_size or cp.shape[1] != args.input_size:
                                cp = cv2.resize(cp, (args.input_size, args.input_size),
                                                interpolation=cv2.INTER_LINEAR)
                        cp = np.clip(cp, -10.0, 10.0) / 5.0
                        p = np.concatenate([p, cp[..., None]], axis=2)
                if args.in_chans == 3:
                    x = p.astype(np.float32) / 255.0
                    x = (x - IMAGENET_MEAN) / IMAGENET_STD
                else:
                    rgb = p[:, :, :3].astype(np.float32) / 255.0
                    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
                    x = np.concatenate([rgb, p[:, :, 3:4].astype(np.float32)], axis=2)
                batch.append(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float())
            tens = torch.stack(batch, 0).to(device)
            with torch.no_grad():
                if args.tta:
                    ps = [torch.softmax(model.predict_only(tens), 1)[:, 1]]
                    for dims in ([3], [2], [2, 3]):
                        ps.append(torch.softmax(model.predict_only(torch.flip(tens, dims)), 1)[:, 1])
                    prob = torch.stack(ps, 0).mean(0)
                else:
                    prob = torch.softmax(model.predict_only(tens), 1)[:, 1]
            for k, lab in enumerate(chunk):
                cy, cx = cents[i + k]
                patches.append(float(prob[k].cpu())); coords.append((float(cx), float(cy)))
                rec_labels.append(int(lab))
        for (cx, cy), p in zip(coords, patches):
            recs.append((base, cx, cy, p))
        print(f"[{hi}/{len(bmps)}] {base}: masks={len(labels)} gt={len(gts)}", flush=True)

    if args.dump_scores:
        out = Path(args.dump_scores)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            base=np.array([r[0] for r in recs]),
            label=np.array(rec_labels, dtype=np.int64),
            cx=np.array([r[1] for r in recs], dtype=np.float32),
            cy=np.array([r[2] for r in recs], dtype=np.float32),
            p=np.array([r[3] for r in recs], dtype=np.float32))
        print(f"写出分数 {out} n={len(recs)}", flush=True)

    print(f"\n总 GT: {total_gt}")
    best = {'f1': -1}; fixed = None
    for thr in np.arange(0.05, 0.9995, 0.005):
        by = {}
        for (base, cx, cy, p) in recs:
            if p >= thr:
                by.setdefault(base, []).append((cx, cy, p))
        tp = fp = fn = 0
        for base, dets in by.items():
            gts, thr_px = gt_by[base]
            t, f, n = match(dets, gts, thr_px); tp += t; fp += f; fn += n
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        if f1 > best['f1']:
            best = dict(f1=f1, thr=thr, tp=tp, fp=fp, fn=fn, p=pr, r=rc)
        if args.fixed_thr is not None and abs(thr - args.fixed_thr) < 1e-6:
            fixed = dict(f1=f1, thr=thr, tp=tp, fp=fp, fn=fn, p=pr, r=rc)
    print(f"oracle: thr={best['thr']:.2f} F1={best['f1']:.4f} P={best['p']:.4f} R={best['r']:.4f} "
          f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    if fixed:
        print(f"fixed {fixed['thr']:.2f}: F1={fixed['f1']:.4f} P={fixed['p']:.4f} R={fixed['r']:.4f} "
              f"TP={fixed['tp']} FP={fixed['fp']} FN={fixed['fn']}")


if __name__ == '__main__':
    main()
