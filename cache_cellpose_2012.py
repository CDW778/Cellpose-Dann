#!/usr/bin/env python3
"""
ICPR2012：对 train/val/test 的 HPF 跑 Cellpose cp=-3 并缓存 mask。
支持断点续跑。输出 {out}/{split}_masks_cp3/{base}_mask.npy
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from cellpose import models, io, core

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -3.0
TILE_NORM_BLOCKSIZE = 0
MIN_SIZE = 15
BATCH_SIZE = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=str(DST))
    ap.add_argument('--cellprob', type=float, default=CELLPROB_THRESHOLD)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = ap.parse_args()

    dst = Path(args.dst)
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}", flush=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=CELLPOSE_WEIGHT)

    for split in args.splits:
        img_dir = dst / "train-val-test" / f"{split}-original-img"
        out_dir = dst / "cellpose_cache" / f"{split}_masks_cp3"
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted(list(img_dir.glob("*.bmp")) + list(img_dir.glob("*.tiff")))
        todo = [p for p in imgs if not (out_dir / f"{p.stem}_mask.npy").exists()]
        print(f"[{split}] 共 {len(imgs)} 张, 待处理 {len(todo)}", flush=True)
        for p in tqdm(todo, desc=split):
            img = io.imread(str(p))
            masks, _, _ = model.eval(img, batch_size=BATCH_SIZE,
                                     flow_threshold=FLOW_THRESHOLD,
                                     cellprob_threshold=args.cellprob,
                                     min_size=MIN_SIZE,
                                     normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE})
            m = masks if isinstance(masks, np.ndarray) else np.array(masks)
            np.save(out_dir / f"{p.stem}_mask.npy", m.astype(np.int32))
        n = len(list(out_dir.glob("*_mask.npy")))
        print(f"[{split}] 完成, 共 {n} 个 mask", flush=True)


if __name__ == "__main__":
    main()
