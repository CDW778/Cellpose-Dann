#!/usr/bin/env python3
"""逐例分析 test 04 的 FN/FP（D seed1，固定阈值）。"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from dann_dataset import _bounds, PATCH
from gen_dataset_2012 import find_csv, load_gt_regions, n_components, label_bboxes

DST = Path('/home/hyw/dataset/chendawei/train-classification/ICPR-2012')
HEAD = 38


def res_um(b):
    return 0.2456 if str(b).startswith('A') else (0.2273 + 0.22753) / 2.0


def match(dets, gts, thr):
    used, g2d = set(), {}
    for gi, (gx, gy) in enumerate(gts):
        cand = sorted((float(np.hypot(d[0] - gx, d[1] - gy)), i)
                      for i, d in enumerate(dets))
        for dist, di in cand:
            if dist > thr:
                break
            if di in used:
                continue
            used.add(di); g2d[gi] = di
            break
    d2g = {v: k for k, v in g2d.items()}
    fp = [i for i in range(len(dets)) if i not in used]
    fn = [gi for gi in range(len(gts)) if gi not in g2d]
    return g2d, d2g, fp, fn


def _fill(ann, binm, color, alpha):
    if binm is not None and binm.any():
        ann[binm > 0] = (ann[binm > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def _contour(ann, binm, color):
    if binm is not None and binm.any():
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, cnts, -1, color, 1)


def panel(raw, mask, regs, target, lab, l, t, r, b, text, cent_map, bb):
    ann = raw.copy()
    H, W = mask.shape
    for gj, (xs, ys) in enumerate(regs):
        xx, yy = xs - l, ys - t
        ok = (xx >= 0) & (xx < PATCH) & (yy >= 0) & (yy < PATCH)
        if not ok.any():
            continue
        binm = np.zeros((PATCH, PATCH), np.uint8)
        binm[yy[ok], xx[ok]] = 1
        color = (0, 200, 0) if gj == target else (255, 255, 0)
        _fill(ann, binm, color, 0.42 if gj == target else 0.28)
        _contour(ann, binm, color)
        gx = int(round(xs[ok].mean())) - l
        gy = int(round(ys[ok].mean())) - t
        if 0 <= gx < PATCH and 0 <= gy < PATCH:
            cv2.drawMarker(ann, (gx, gy), color, cv2.MARKER_CROSS, 11, 1)
            if gj != target:
                cv2.putText(ann, f'gt{gj}', (gx + 3, gy - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA)
    if lab is not None and lab in cent_map and lab in bb:
        mbin = (mask[t:b, l:r] == lab).astype(np.uint8)
        _fill(ann, mbin, (0, 0, 200), 0.22)
        _contour(ann, mbin, (0, 0, 255))
        cy, cx = cent_map[lab]
        mcx, mcy = int(round(cx - l)), int(round(cy - t))
        if 0 <= mcx < PATCH and 0 <= mcy < PATCH:
            cv2.drawMarker(ann, (mcx, mcy), (255, 0, 0), cv2.MARKER_CROSS, 12, 1)
        y0, y1, x0, x1 = bb[lab]
        bcx, bcy = int((x0 + x1) / 2 - l), int((y0 + y1) / 2 - t)
        if 0 <= bcx < PATCH and 0 <= bcy < PATCH:
            cv2.circle(ann, (bcx, bcy), 3, (0, 220, 255), -1)
    for i, line in enumerate(text):
        cv2.putText(ann, line, (3, 14 + i * 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.full((PATCH + HEAD, PATCH * 2 + 2, 3), 255, np.uint8)
    canvas[HEAD:, :PATCH] = raw
    canvas[HEAD:, PATCH + 2:] = ann
    cv2.putText(canvas, text[0], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--thr', type=float, default=0.92)
    ap.add_argument('--compare-thr', type=float, default=0.40)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--mask-dir', default=None)
    ap.add_argument('--img-dir', default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.mask_dir else DST / 'cellpose_cache' / f'{args.split}_masks_cp3'
    img_dir = Path(args.img_dir) if args.img_dir else DST / 'train-val-test' / f'{args.split}-original-img'

    d = np.load(args.scores, allow_pickle=True)
    by = {}
    for b, lab, cx, cy, p in zip(d['base'], d['label'], d['cx'], d['cy'], d['p']):
        by.setdefault(str(b), []).append((int(lab), float(cx), float(cy), float(p)))

    rows = []
    coverage = {}
    for base in sorted(by):
        img = cv2.cvtColor(cv2.imread(str(img_dir / f'{base}.bmp')), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_dir / f'{base}_mask.npy')
        labels = [int(x) for x in np.unique(mask) if x != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        cent_map = {int(l): (float(c[0]), float(c[1])) for l, c in zip(labels, cents)}
        bb = label_bboxes(mask)
        regs = load_gt_regions(find_csv(base))
        gts = [(float(xs.mean()), float(ys.mean())) for xs, ys in regs]
        thr_px = 5.0 / res_um(base)

        recs = by[base]
        smap = {lab: (cx, cy, p) for lab, cx, cy, p in recs}
        dets = [(cx, cy, p, lab) for lab, cx, cy, p in recs if p >= args.thr]
        g2d, d2g, fp_idx, fn_gt = match(dets, gts, thr_px)
        g2d_cmp, _, _, _ = match([(cx, cy, p, lab) for lab, cx, cy, p in recs
                                  if p >= args.compare_thr], gts, thr_px)
        lost_tp = {gi for gi in g2d_cmp if gi not in g2d}

        cov = 0
        for gi, (gx, gy) in enumerate(gts):
            if any(np.hypot(cx - gx, cy - gy) <= thr_px for cy, cx in cent_map.values()):
                cov += 1
        coverage[base] = (cov, len(gts))

        for gi in fn_gt:
            xs, ys = regs[gi]
            gx, gy = float(xs.mean()), float(ys.mean())
            gta = int(len(xs))
            comp = n_components(xs, ys)
            labs = [int(l) for l in np.unique(mask[ys, xs]) if l != 0]
            best = (0.0, None, 0)
            for lab in labs:
                inter = int(np.sum(mask[ys, xs] == lab))
                ma = int(np.sum(mask == lab))
                iou = inter / (gta + ma - inter)
                if iou > best[0]:
                    best = (iou, lab, ma)
            iou, lab, ma = best
            if lab is None:
                near = min(((float(np.hypot(c[1] - gx, c[0] - gy)), int(l))
                            for l, c in cent_map.items()), default=(1e9, None))
                if near[0] <= thr_px:
                    lab = near[1]; ma = int(np.sum(mask == lab))
                else:
                    lab = None; ma = 0
            p = smap[lab][2] if lab in smap else -1.0
            if lab in smap:
                cdist = float(np.hypot(smap[lab][0] - gx, smap[lab][1] - gy) * res_um(base))
            else:
                cdist = -1.0

            if lab is None:
                cx0, cy0 = gx, gy
            else:
                cy0, cx0 = cent_map[lab]
            l, t, r, b = _bounds(int(round(cx0)), int(round(cy0)), PATCH, *mask.shape)
            ids_in = []
            for gj, (gxs, gys) in enumerate(regs):
                xx, yy = gxs - l, gys - t
                ok = (xx >= 0) & (xx < PATCH) & (yy >= 0) & (yy < PATCH)
                if ok.any():
                    ids_in.append(gj)
            extra = [g for g in ids_in if g != gi]

            competition = False
            for di, (cx, cy, pp, ll) in enumerate(dets):
                if di in d2g and d2g[di] != gi and np.hypot(cx - gx, cy - gy) <= thr_px:
                    competition = True
                    break

            flags = []
            if lab is None:
                cat = 'candidate_missing'
            elif iou < 0.3 or (gta > 0 and ma > 3 * gta):
                cat = 'mask_merge'
            elif competition or len(ids_in) > 1:
                cat = 'multi_gt'
            elif p < 0.05:
                cat = 'classifier_hard_fn'
            elif p < args.thr:
                cat = 'boundary'
            elif comp >= 2:
                cat = 'telophase'
            else:
                cat = 'other'
            if lab is not None and iou < 0.3:
                flags.append('iou_lt_0.3')
            if gta > 0 and ma > 3 * gta:
                flags.append('area_gt_3x')
            if p < 0.05:
                flags.append('p_lt_0.05')
            if 0.05 <= p < args.thr:
                flags.append('boundary')
            if competition:
                flags.append('matching_competition')
            if len(ids_in) > 1:
                flags.append('multi_gt_crop')
            if comp >= 2:
                flags.append('ncomp_ge_2')
            if gi in lost_tp:
                flags.append('lost_tp_0.40')

            raw = img[t:b, l:r]
            if raw.shape[0] != PATCH or raw.shape[1] != PATCH:
                raw = cv2.resize(raw, (PATCH, PATCH))
            text = [f'{base} gt{gi} {cat}',
                    f'p={p:.2f} IoU={iou:.2f} cdist={cdist:.1f}um',
                    f'GTa={gta} Ma={ma} ncomp={comp}',
                    f'GTs={",".join(f"gt{x}" for x in ids_in)}']
            canvas = panel(raw, mask, regs, gi, lab, l, t, r, b, text, cent_map, bb)
            fname = f'FN_{len(rows)+1:03d}_{base}_gt{gi}_{cat}_p{p:.2f}.png'
            cv2.imwrite(str(out / 'images' / fname), canvas)
            rows.append(dict(kind='FN', base=base, gt=gi, label=lab, category=cat,
                             flags=';'.join(flags), p=round(p, 4), iou=round(iou, 3),
                             gt_area=gta, mask_area=ma, centroid_dist_um=round(cdist, 2),
                             ncomp=comp, n_gt_in_crop=len(ids_in),
                             extra_gt_in_crop=';'.join(str(x) for x in extra),
                             is_lost_tp=int(gi in lost_tp), file=fname))

        for di in fp_idx:
            cx, cy, p, lab = dets[di]
            if lab in cent_map:
                cy0, cx0 = cent_map[lab]
            else:
                cy0, cx0 = cy, cx
            best_dist, best_gi = 1e9, None
            for gi, (gx, gy) in enumerate(gts):
                dd = float(np.hypot(cx - gx, cy - gy) * res_um(base))
                if dd < best_dist:
                    best_dist, best_gi = dd, gi
            if best_gi is not None and np.hypot(cx - gts[best_gi][0], cy - gts[best_gi][1]) <= thr_px:
                cat = 'duplicate_near_gt'
            else:
                cat = 'background_fp'
            flags = ['high_conf' if p >= 0.98 else 'boundary_fp']
            l, t, r, b = _bounds(int(round(cx0)), int(round(cy0)), PATCH, *mask.shape)
            ids_in = []
            for gj, (gxs, gys) in enumerate(regs):
                xx, yy = gxs - l, gys - t
                ok = (xx >= 0) & (xx < PATCH) & (yy >= 0) & (yy < PATCH)
                if ok.any():
                    ids_in.append(gj)
            raw = img[t:b, l:r]
            if raw.shape[0] != PATCH or raw.shape[1] != PATCH:
                raw = cv2.resize(raw, (PATCH, PATCH))
            text = [f'{base} lab{lab} {cat}',
                    f'p={p:.2f} nearest_gt={best_dist:.1f}um',
                    f'mask_area={int(np.sum(mask==lab)) if lab in cent_map else 0}',
                    f'GTs={",".join(f"gt{x}" for x in ids_in)}']
            canvas = panel(raw, mask, regs, -1, lab, l, t, r, b, text, cent_map, bb)
            fname = f'FP_{len(rows)+1:03d}_{base}_lab{lab}_{cat}_p{p:.2f}.png'
            cv2.imwrite(str(out / 'images' / fname), canvas)
            rows.append(dict(kind='FP', base=base, gt=-1, label=lab, category=cat,
                             flags=';'.join(flags), p=round(p, 4), iou=0,
                             gt_area=0, mask_area=int(np.sum(mask == lab)) if lab in cent_map else 0,
                             centroid_dist_um=round(best_dist, 2), ncomp=0,
                             n_gt_in_crop=len(ids_in),
                             extra_gt_in_crop=';'.join(str(x) for x in ids_in),
                             is_lost_tp=0, file=fname))

    fields = list(rows[0].keys()) if rows else []
    with open(out / 'cases.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    fn_counts = Counter(r['category'] for r in rows if r['kind'] == 'FN')
    fp_counts = Counter(r['category'] for r in rows if r['kind'] == 'FP')
    lost = [r for r in rows if r['kind'] == 'FN' and r['is_lost_tp']]
    with open(out / 'summary.md', 'w') as f:
        f.write(f'# FN/FP summary\n\n')
        f.write(f'- threshold: {args.thr}\n')
        f.write(f'- compare threshold: {args.compare_thr}\n')
        f.write(f'- FN: {sum(fn_counts.values())}\n')
        f.write(f'- FP: {sum(fp_counts.values())}\n')
        f.write(f'- lost TP (0.40 -> 0.92): {len(lost)}\n\n')
        f.write('## FN categories\n')
        for k, v in fn_counts.most_common():
            f.write(f'- {k}: {v}\n')
        f.write('\n## FP categories\n')
        for k, v in fp_counts.most_common():
            f.write(f'- {k}: {v}\n')
        f.write('\n## Candidate coverage\n')
        for b, (c, n) in sorted(coverage.items()):
            f.write(f'- {b}: {c}/{n}\n')
        f.write('\n## Lost TP\n')
        for r in lost:
            f.write(f"- {r['base']} gt{r['gt']} p={r['p']} {r['category']} {r['file']}\n")

    print(f'FN={sum(fn_counts.values())} FP={sum(fp_counts.values())} lost_tp={len(lost)}')
    print('FN', dict(fn_counts))
    print('FP', dict(fp_counts))
    print('saved', out)


if __name__ == '__main__':
    main()
