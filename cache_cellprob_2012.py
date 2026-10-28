#!/usr/bin/env python3
"""缓存 ICPR2012 train/val/test 的 Cellpose cellprob（flows[2]），用于 DANN 第 4 通道。"""
import time
from pathlib import Path

import cv2
import numpy as np

from cellpose import models, core

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW = 1.0
CELLPROB = -3.0
MIN_SIZE = 15
TILE_NORM = 0
TILE_OVERLAP = 0.1
BATCH = 16
SPLITS = ["train", "val", "test"]


def main():
    gpu = core.use_gpu()
    print(f"GPU={gpu}; loading Cellpose ...", flush=True)
    cp = models.CellposeModel(gpu=gpu, pretrained_model=CELLPOSE_WEIGHT)
    print("model loaded", flush=True)
    for split in SPLITS:
        img_dir = DST / "train-val-test" / f"{split}-original-img"
        out_dir = DST / "cellprob_cache" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        bmps = sorted(img_dir.glob("*.bmp"))
        print(f"\n[{split}] {len(bmps)} images", flush=True)
        for i, bmp in enumerate(bmps, 1):
            base = bmp.stem
            op = out_dir / f"{base}_cellprob.npy"
            if op.exists():
                print(f"[{split}] {i}/{len(bmps)} {base}: cached", flush=True)
                continue
            t0 = time.time()
            img = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
            masks, flows, styles = cp.eval(img, batch_size=BATCH,
                                           flow_threshold=FLOW,
                                           cellprob_threshold=CELLPROB,
                                           min_size=MIN_SIZE,
                                           normalize={"tile_norm_blocksize": TILE_NORM},
                                           tile_overlap=TILE_OVERLAP, augment=False)
            cellprob = np.asarray(flows[2], dtype=np.float16)
            np.save(op, cellprob)
            print(f"[{split}] {i}/{len(bmps)} {base}: {cellprob.shape} {time.time()-t0:.1f}s", flush=True)
    print("\ncellprob cache done", flush=True)


if __name__ == "__main__":
    main()
