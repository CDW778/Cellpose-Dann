#!/usr/bin/env python3
"""用当前最优模型给 train 正样本打分，挖出低分"难正样本"索引。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dann_model import DANNModel
from dann_dataset import CandidateDataset

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT = Path("/home/hyw/chendawei/dann_finetune")
NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', default=str(OUT / "candidates_train.npz"))
    ap.add_argument('--weight', default=str(OUT / "best_s3_seed0.pth"))
    ap.add_argument('--out', default=str(OUT / "hard_pos_idx.npy"))
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    data = dict(np.load(args.cand, allow_pickle=True))
    pos = np.where(data['y'] == 1)[0]
    root = DST / "train-val-test" / "train-original-img"
    ds = CandidateDataset('train', data, pos, root, augment=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DANNModel(2, NDC, 0.0)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval().to(device)

    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            lg = model.predict_only(xb.to(device))
            probs.extend(torch.softmax(lg, 1)[:, 1].cpu().numpy().tolist())
    probs = np.array(probs)

    print(f"score {args.weight} on {len(pos)} train positives")
    for q in [0, 10, 25, 50, 75, 90, 100]:
        print(f"  p{q:3d} = {np.percentile(probs, q):.4f}")
    for t in [0.1, 0.3, 0.5, 0.8, 0.9]:
        print(f"  p < {t:.1f}: {int((probs < t).sum())} / {len(probs)}")

    hard = pos[probs < args.thr]
    np.save(args.out, hard)
    np.save(str(OUT / "train_pos_scores.npy"), np.stack([pos.astype(np.float32), probs.astype(np.float32)]))
    print(f"\n难正(thr<{args.thr}): {len(hard)} 个 -> {args.out}")
    print(f"分数记录 -> {OUT/'train_pos_scores.npy'}")


if __name__ == "__main__":
    main()
