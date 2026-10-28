#!/usr/bin/env python3
"""构建 ICPR2012 slide-level 4-fold CV 候选，并集中图片软链接。"""
import os
import numpy as np
from pathlib import Path

OUT = Path("/home/hyw/chendawei/dann_finetune")
CV = OUT / "cv"
DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
TRAIN_IMG = DST / "train-val-test" / "train-original-img"
VAL_IMG = DST / "train-val-test" / "val-original-img"
KEYS = ["base", "label", "x0", "y0", "x1", "y1", "y", "cx", "cy"]


def load_cands():
    tr = dict(np.load(OUT / "candidates_train.npz", allow_pickle=True))
    va = dict(np.load(OUT / "candidates_val.npz", allow_pickle=True))
    return {k: np.concatenate([tr[k], va[k]], axis=0) for k in KEYS}


def subset(data, bases_mask):
    return {k: data[k][bases_mask] for k in KEYS}


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def main():
    CV.mkdir(parents=True, exist_ok=True)
    data = load_cands()
    slides = np.array([str(b)[1:3] for b in data["base"]])
    print("总候选:", len(slides), "slides:", sorted(set(slides)), flush=True)

    # 图片集中：images_all 软链接
    img_all = CV / "images_all"
    img_all.mkdir(parents=True, exist_ok=True)
    for b in sorted(set(str(x) for x in data["base"])):
        link = img_all / f"{b}.bmp"
        if link.is_symlink() or link.exists():
            continue
        src = TRAIN_IMG / f"{b}.bmp"
        if not src.exists():
            src = VAL_IMG / f"{b}.bmp"
        if not src.exists():
            print("缺少图片:", b, flush=True)
            continue
        os.symlink(str(src), str(link))
    print("images_all:", len(list(img_all.glob('*.bmp'))), flush=True)

    for f in ["00", "01", "02", "03"]:
        tr = subset(data, slides != f)
        va = subset(data, slides == f)
        d = CV / f"fold{int(f)}"
        save(tr, d / "candidates_train.npz")
        save(va, d / "candidates_val.npz")
        print(f"fold{int(f)}: train={len(tr['y'])} (pos {int(tr['y'].sum())}) "
              f"val={len(va['y'])} (pos {int(va['y'].sum())})", flush=True)

    # final: 00-03 全部作为 train
    save(data, CV / "final" / "candidates_train.npz")
    print(f"final: train={len(data['y'])} (pos {int(data['y'].sum())})", flush=True)


if __name__ == '__main__':
    main()
