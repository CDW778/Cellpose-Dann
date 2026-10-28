#!/usr/bin/env python3
"""对候选 npz 逐样本打分，导出 OOF/val 预测（支持 3ch 或 mask 4ch）。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dann_model import DANNModel
from dann_dataset import CandidateDataset

NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', required=True)
    ap.add_argument('--cand', required=True)
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--mask-dir', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--in-chans', type=int, default=3)
    ap.add_argument('--mask-channel', action='store_true')
    ap.add_argument('--crop-center', choices=['bbox', 'centroid'], default='bbox')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    cand = dict(np.load(args.cand, allow_pickle=True))
    idx = np.arange(len(cand['y']))
    ds = CandidateDataset('val', cand, idx, Path(args.img_root), augment=False,
                          crop_size=args.input_size, in_chans=args.in_chans,
                          mask_dir=args.mask_dir, crop_center=args.crop_center,
                          mask_channel=args.mask_channel)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    persistent_workers=args.workers > 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DANNModel(2, NDC, 0.0, in_chans=args.in_chans)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval().to(device)

    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            lg = model.predict_only(xb.to(device))
            probs.extend(torch.softmax(lg, dim=1)[:, 1].cpu().numpy().tolist())
    probs = np.asarray(probs, dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, base=cand['base'], label=cand['label'],
                        cx=cand['cx'], cy=cand['cy'], y=cand['y'], p=probs)
    print(f"saved {out} n={len(probs)} mean={probs.mean():.4f} max={probs.max():.4f}", flush=True)


if __name__ == '__main__':
    main()
