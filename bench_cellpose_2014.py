#!/usr/bin/env python3
"""Cellpose 速度基准：batch_size 与 tile_overlap（官方 2014 test 的 A/H 各 1 帧）。"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import models, io, core

ROOT = Path("/home/hyw/dataset/chendawei/ICPR2014")
A_FRAME = ROOT / "mitos_atypia_2014_testing_aperio/A06/frames/x40/A06_00Aa.tiff"
H_FRAME = ROOT / "mitos_atypia_2014_testing_hamamatsu/H06/frames/x40/H06_00Aa.tiff"
OUT = Path("/home/hyw/chendawei/icpr2014_eval")
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW = 1.0
CELLPROB = -3.0
MIN_SIZE = 15
TILE_NORM = 0


def run(cp, img, batch, overlap):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    masks, _, _ = cp.eval(img, batch_size=batch, flow_threshold=FLOW,
                          cellprob_threshold=CELLPROB, min_size=MIN_SIZE,
                          normalize={"tile_norm_blocksize": TILE_NORM},
                          tile_overlap=overlap, augment=False)
    dt = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 1e9
    m = masks if isinstance(masks, np.ndarray) else np.array(masks)
    return m, dt, mem


def compare(m1, m2):
    eq = bool(np.array_equal(m1, m2))
    fg1 = m1 > 0; fg2 = m2 > 0
    inter = float((fg1 & fg2).sum()); union = float((fg1 | fg2).sum())
    fg_iou = inter / union if union else 1.0
    # 逐 label 平均 IoU（以 m1 的 label 为准）
    K = int(max(m1.max(), m2.max())) + 1
    ct = np.bincount((m1.astype(np.int64) * K + m2).ravel(), minlength=K * K).reshape(K, K)
    ious = []
    for l in range(1, K):
        cnt = int(ct[l].sum())
        if cnt == 0:
            continue
        j = int(ct[l].argmax()); inter_l = int(ct[l, j])
        union_l = cnt + int(ct[:, j].sum()) - inter_l
        if union_l > 0:
            ious.append(inter_l / union_l)
    return eq, fg_iou, float(np.mean(ious)) if ious else 1.0, int(m1.max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    gpu = core.use_gpu()
    print(f"GPU={gpu}; loading model ...", flush=True)
    cp = models.CellposeModel(gpu=gpu, pretrained_model=CELLPOSE_WEIGHT)
    print(f"model loaded\n", flush=True)

    rows = []
    for tag, path in [("A06_00Aa", A_FRAME), ("H06_00Aa", H_FRAME)]:
        img = io.imread(str(path))
        print(f"\n=== {tag} shape={img.shape} ===", flush=True)
        # warmup
        _ = run(cp, img, 1, 0.1)
        base, dt0, mem0 = run(cp, img, 1, 0.1)
        print(f"baseline batch=1 overlap=0.1: {dt0:.2f}s mem={mem0:.2f}GB masks={int(base.max())}", flush=True)
        rows.append(dict(frame=tag, batch=1, overlap=0.1, time_s=round(dt0, 2),
                         mem_gb=round(mem0, 2), n_masks=int(base.max()), same=True,
                         fg_iou=1.0, label_iou=1.0))

        best = (dt0, 1)
        for b in [4, 8, 16, 32]:
            try:
                m, dt, mem = run(cp, img, b, 0.1)
                eq, fg, li, nm = compare(base, m)
                print(f"batch={b:2d} overlap=0.1: {dt:6.2f}s mem={mem:5.2f}GB masks={nm} "
                      f"same={eq} fgIoU={fg:.4f} labelIoU={li:.4f}", flush=True)
                rows.append(dict(frame=tag, batch=b, overlap=0.1, time_s=round(dt, 2),
                                 mem_gb=round(mem, 2), n_masks=nm, same=eq,
                                 fg_iou=round(fg, 5), label_iou=round(li, 5)))
                if dt < best[0]:
                    best = (dt, b)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                print(f"batch={b:2d} OOM/ERR: {type(e).__name__}", flush=True)
                rows.append(dict(frame=tag, batch=b, overlap=0.1, time_s=-1,
                                 mem_gb=-1, n_masks=-1, same=False, fg_iou=-1, label_iou=-1))
        print(f"best batch (overlap=0.1): {best[1]} ({best[0]:.2f}s)", flush=True)

        for ov in [0.05, 0.0]:
            try:
                m, dt, mem = run(cp, img, best[1], ov)
                eq, fg, li, nm = compare(base, m)
                print(f"batch={best[1]:2d} overlap={ov:.2f}: {dt:6.2f}s mem={mem:5.2f}GB masks={nm} "
                      f"same={eq} fgIoU={fg:.4f} labelIoU={li:.4f}", flush=True)
                rows.append(dict(frame=tag, batch=best[1], overlap=ov, time_s=round(dt, 2),
                                 mem_gb=round(mem, 2), n_masks=nm, same=eq,
                                 fg_iou=round(fg, 5), label_iou=round(li, 5)))
            except RuntimeError as e:
                torch.cuda.empty_cache()
                print(f"overlap={ov} ERR: {type(e).__name__}", flush=True)

    with open(OUT / "cellpose_speed_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n写出: {OUT/'cellpose_speed_bench.csv'}", flush=True)


if __name__ == "__main__":
    main()
