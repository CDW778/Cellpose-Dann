#!/usr/bin/env python3
"""用某折 best_s3 对折内 val 候选打分，导出 OOF 预测。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dann_model import DANNModel, load_pretrained_dann
from dann_dataset import CandidateDataset

NUM_DOMAIN_CLASSES = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', required=True)
    ap.add_argument('--cand', required=True)
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--input-size', type=int, default=224)
    args = ap.parse_args()

    cand = dict(np.load(args.cand, allow_pickle=True))
    idx = np.arange(len(cand['y']))
    ds = CandidateDataset('val', cand, idx, Path(args.img_root), augment=False,
                          crop_size=args.input_size, in_chans=3)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    persistent_workers=args.workers > 0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DANNModel(num_classes=2, num_domain_classes=NUM_DOMAIN_CLASSES, lambda_val=0.0)
    miss, unexp = load_pretrained_dann(model, args.weight)
    print(f"weight={args.weight} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.to(device).eval()

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
    print(f"saved {out} n={len(probs)} mean p={probs.mean():.4f} max={probs.max():.4f}", flush=True)


if __name__ == '__main__':
    main()
