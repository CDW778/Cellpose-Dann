#!/usr/bin/env python3
"""汇总 4 折 OOF，扫描一个全局阈值，并给出中位 best epoch。"""
import argparse
from pathlib import Path

import numpy as np

from eval_2012_dann_live import match, load_gt_points, res_um
from gen_dataset_2012 import find_csv


def load_fold(cv, f):
    p = cv / f"fold{f}" / "val_preds.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    return dict(np.load(p, allow_pickle=True))


def gt_for(base):
    csv_p = find_csv(base)
    pts = load_gt_points(csv_p) if csv_p else []
    return pts, 5.0 / res_um(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-dir', default='/home/hyw/chendawei/dann_finetune/cv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cv = Path(args.cv_dir)
    out_csv = Path(args.out) if args.out else cv / "cv_pooled_summary.csv"

    folds = {}
    for f in range(4):
        d = load_fold(cv, f)
        folds[f] = d
        print(f"fold{f}: n={len(d['p'])} mean_p={d['p'].mean():.4f} max={d['p'].max():.4f}", flush=True)

    gt_cache = {}
    def gt(b):
        if b not in gt_cache:
            gt_cache[b] = gt_for(b)
        return gt_cache[b]

    rows = []
    best = None
    for thr in np.arange(0.05, 0.995, 0.01):
        per_fold = {}
        tp = fp = fn = 0
        for f, d in folds.items():
            t = fp_f = n = 0
            by = {}
            for i, b in enumerate(d['base']):
                by.setdefault(str(b), []).append((float(d['cx'][i]), float(d['cy'][i]), float(d['p'][i])))
            for b, recs in by.items():
                pts, thr_px = gt(b)
                dets = [(x, y, p) for (x, y, p) in recs if p >= thr]
                a, c, e = match(dets, pts, thr_px)
                t += a; fp_f += c; n += e
            per_fold[f] = (t, fp_f, n)
            tp += t; fp += fp_f; fn += n
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        rows.append((thr, tp, fp, fn, pr, rc, f1))
        if best is None or f1 > best[0]:
            best = (f1, thr, tp, fp, fn, pr, rc, per_fold)

    f1, thr, tp, fp, fn, pr, rc, per_fold = best
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w') as fh:
        fh.write("thr,tp,fp,fn,precision,recall,f1\n")
        for r in rows:
            fh.write(f"{r[0]:.2f},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")
    (cv / "cv_best_thr.txt").write_text(f"{thr:.2f}\n")

    epochs = []
    for f in range(4):
        p = cv / f"fold{f}" / "best_epoch_s3.txt"
        if p.exists():
            epochs.append(int(p.read_text().strip()))
    med = int(round(float(np.median(epochs)))) if epochs else 80
    (cv / "cv_median_epoch.txt").write_text(f"{med}\n")

    print("\n===== pooled OOF best =====", flush=True)
    print(f"global thr={thr:.2f} F1={f1:.4f} P={pr:.4f} R={rc:.4f} TP={tp} FP={fp} FN={fn}", flush=True)
    for f, (t, c, e) in sorted(per_fold.items()):
        p_ = t / (t + c) if (t + c) else 0.0
        r_ = t / (t + e) if (t + e) else 0.0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        print(f"  fold{f}: F1={f1_:.4f} P={p_:.4f} R={r_:.4f} TP={t} FP={c} FN={e}", flush=True)
    print(f"best epochs={epochs} median={med}", flush=True)
    print(f"saved {out_csv}; thr -> {cv/'cv_best_thr.txt'}; epoch -> {cv/'cv_median_epoch.txt'}", flush=True)


if __name__ == '__main__':
    main()
