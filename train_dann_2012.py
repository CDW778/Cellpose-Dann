#!/usr/bin/env python3
"""在 ICPR2012 Cellpose 候选上微调 DANN（分阶段）。

阶段 1：冻结 backbone，只训 mitosis_classifier。
阶段 2：解冻 layer4 + 头，低 LR。

采样：每正样本配 --neg-per-pos 个负样本；加权 CE class_weight=[1, neg_per_pos]
（等效 1:1）。只用有丝分裂损失，域头不参与。选模型按 val(03) e2e 5µm F1。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dann_model import DANNModel, load_pretrained_dann
from dann_dataset import CandidateDataset
from eval_2012_dann_live import match, load_gt_points, res_um
from gen_dataset_2012 import find_csv

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
NUM_DOMAIN_CLASSES = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
MIDOG_WEIGHT = "/home/hyw/chendawei/classification-weight/best_dann_model.pth"


class SoftmaxFocalLoss(nn.Module):
    """多类 softmax Focal Loss：FL = -alpha_t * (1-p_t)^gamma * log(p_t)。"""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        alpha_t = self.alpha[target]
        loss = -(alpha_t * (1.0 - pt).pow(self.gamma) * logpt)
        return loss.mean()


def build_gt(cand):
    gt = {}
    for base in sorted(set(str(b) for b in cand['base'])):
        csv_p = find_csv(base)
        pts = load_gt_points(csv_p) if csv_p else []
        gt[base] = (pts, 5.0 / res_um(base))
    return gt


def make_epoch_index(y, neg_per_pos, rng, hard_pos=None, hard_mult=1):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    idx = list(pos)
    if hard_pos is not None and len(hard_pos) > 0 and hard_mult > 1:
        idx += list(hard_pos) * (hard_mult - 1)
    if len(pos) and len(neg):
        idx += list(rng.choice(neg, size=len(pos) * neg_per_pos, replace=True))
    rng.shuffle(idx)
    return np.array(idx, dtype=np.int64)


def loader(ds, batch, workers, shuffle=False):
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kw)


def compute_features(model, cand, index, img_root, device, batch, workers, crop_size=224, in_chans=3, cellprob_dir=None,
                     mask_dir=None, crop_center='bbox', soft_mask=False, mask_sigma=4.0,
                     mask_dilate=4, mask_bg_sigma=20.0, mask_channel=False):
    ds = CandidateDataset('x', cand, index, img_root, augment=False, crop_size=crop_size,
                          in_chans=in_chans, cellprob_dir=cellprob_dir, mask_dir=mask_dir,
                          crop_center=crop_center, soft_mask=soft_mask,
                          mask_sigma=mask_sigma, mask_dilate=mask_dilate,
                          mask_bg_sigma=mask_bg_sigma, mask_channel=mask_channel)
    dl = loader(ds, batch, workers, shuffle=False)
    feats = []
    model.eval()
    with torch.no_grad():
        for xb, _ in dl:
            feats.append(model.get_features(xb.to(device)).cpu().numpy())
    return np.concatenate(feats, 0)


def eval_from_features(model, feats, cand, index, gt, device, chunk=8192):
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(feats), chunk):
            lg = model.mitosis_classifier(torch.from_numpy(feats[i:i + chunk]).to(device))
            scores.extend(torch.softmax(lg, dim=1)[:, 1].cpu().numpy().tolist())
    by = {}
    for k, j in enumerate(index):
        base = str(cand['base'][j])
        by.setdefault(base, []).append((float(cand['cx'][j]), float(cand['cy'][j]), scores[k]))
    best = {'f1': -1}
    fixed = None
    for thr in np.arange(0.05, 0.995, 0.01):
        tp = fp = fn = 0
        for base, recs in by.items():
            pts, thr_px = gt[base]
            dets = [(x, y, p) for (x, y, p) in recs if p >= thr]
            t, f, n = match(dets, pts, thr_px)
            tp += t; fp += f; fn += n
        pr = tp / (tp + fp) if (tp + fp) else 0
        rc = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
        if f1 > best['f1']:
            best = {'f1': f1, 'thr': thr, 'tp': tp, 'fp': fp, 'fn': fn, 'p': pr, 'r': rc}
        if abs(thr - 0.5) < 1e-9:
            fixed = {'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'p': pr, 'r': rc}
    return best, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=[1, 2, 3], required=True)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--neg-per-pos', type=int, default=20)
    ap.add_argument('--lr-head', type=float, default=None)
    ap.add_argument('--lr-backbone', type=float, default=1e-5, help='layer4 学习率')
    ap.add_argument('--lr-layer3', type=float, default=5e-6)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--loss', choices=['ce', 'focal'], default='ce')
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--label-smoothing', type=float, default=0.0)
    ap.add_argument('--strong-aug', action='store_true')
    ap.add_argument('--clip-grad', type=float, default=0.0, help='>0 时做梯度裁剪')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default="/home/hyw/chendawei/dann_finetune")
    ap.add_argument('--init', default=None, help='初始化权重；stage2/3 默认取上一阶段最优')
    ap.add_argument('--eval-every', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='', help='输出文件名后缀，如 _seed0')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--in-chans', type=int, default=3)
    ap.add_argument('--cellprob-dir', default=str(DST / 'cellprob_cache'))
    ap.add_argument('--mask-dir-train', default=None)
    ap.add_argument('--mask-dir-val', default=None)
    ap.add_argument('--crop-center', choices=['bbox', 'centroid'], default='bbox')
    ap.add_argument('--soft-mask', action='store_true')
    ap.add_argument('--mask-channel', action='store_true', help='in_chans=4 时第4通道用目标mask')
    ap.add_argument('--mask-sigma', type=float, default=4.0)
    ap.add_argument('--mask-dilate', type=int, default=4)
    ap.add_argument('--mask-bg-sigma', type=float, default=20.0)
    ap.add_argument('--use-gem', action='store_true')
    ap.add_argument('--use-cbam', action='store_true')
    ap.add_argument('--use-od-gate', action='store_true')
    ap.add_argument('--hard-pos-file', default=None, help='难正样本全局索引 .npy')
    ap.add_argument('--hard-pos-mult', type=int, default=3, help='难正样本重复倍数')
    ap.add_argument('--patience', type=int, default=25, help='val 连续不提升的 early-stop 轮数')
    ap.add_argument('--no-val', action='store_true', help='不加载 val、不做评估/选择，只按固定 epoch 训练')
    ap.add_argument('--img-root', default=None, help='覆盖训练/验证图片目录（CV 用 images_all）')
    args = ap.parse_args()

    if args.epochs is None:
        args.epochs = {1: 40, 2: 20, 3: 120}[args.stage]
    if args.batch is None:
        args.batch = 64 if args.stage == 1 else 32
    if args.lr_head is None:
        args.lr_head = {1: 1e-3, 2: 1e-4, 3: 5e-5}[args.stage]
    if args.eval_every is None:
        args.eval_every = 2 if args.stage == 1 else 5

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    tr = dict(np.load(out / "candidates_train.npz", allow_pickle=True))
    if args.no_val:
        va = None
    else:
        va = dict(np.load(out / "candidates_val.npz", allow_pickle=True))
    hard_pos = None
    if args.hard_pos_file:
        hp = np.load(args.hard_pos_file).astype(np.int64)
        hp = hp[tr['y'][hp] == 1]
        hard_pos = hp
    train_root = DST / "train-val-test" / "train-original-img"
    val_root = DST / "train-val-test" / "val-original-img"
    if args.img_root:
        train_root = Path(args.img_root)
        val_root = Path(args.img_root)
    train_mask_dir = Path(args.mask_dir_train) if args.mask_dir_train else DST / "cellpose_cache" / "train_masks_cp3"
    val_mask_dir = Path(args.mask_dir_val) if args.mask_dir_val else DST / "cellpose_cache" / "val_masks_cp3"
    if args.no_val:
        gt_val = None
        val_index = None
    else:
        gt_val = build_gt(va)
        val_index = np.arange(len(va['y']))

    init = args.init
    if init is None:
        if args.stage == 1:
            init = MIDOG_WEIGHT
        elif args.stage == 2:
            init = str(out / ("last_s1.pth" if args.no_val else "best_s1.pth"))
        else:
            init = str(out / ("last_s2.pth" if args.no_val else "best_s2.pth"))
    model = DANNModel(num_classes=2, num_domain_classes=NUM_DOMAIN_CLASSES, lambda_val=0.0,
                      in_chans=args.in_chans, use_gem=args.use_gem,
                      use_cbam=args.use_cbam, use_od_gate=args.use_od_gate)
    missing, unexpected = load_pretrained_dann(model, init)
    print(f"init={init} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.mitosis_classifier.parameters():
        p.requires_grad = True
    groups = [{'params': list(model.mitosis_classifier.parameters()), 'lr': args.lr_head}]
    if args.stage >= 2:
        for p in model.layer4.parameters():
            p.requires_grad = True
        groups.append({'params': list(model.layer4.parameters()), 'lr': args.lr_backbone})
    if args.stage == 3:
        for p in model.layer3.parameters():
            p.requires_grad = True
        groups.append({'params': list(model.layer3.parameters()), 'lr': args.lr_layer3})
    if args.use_cbam:
        for p in model.cbam4.parameters():
            p.requires_grad = True
        groups.append({'params': list(model.cbam4.parameters()), 'lr': args.lr_head})
    if args.use_od_gate:
        for p in model.od_gate4.parameters():
            p.requires_grad = True
        groups.append({'params': list(model.od_gate4.parameters()), 'lr': args.lr_head})
    if args.in_chans > 3:
        model.stem[0].weight.requires_grad = True
        groups.append({'params': [model.stem[0].weight], 'lr': args.lr_backbone})

    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    alpha = torch.tensor([1.0, float(args.neg_per_pos)], device=device)
    if args.loss == 'focal':
        crit = SoftmaxFocalLoss(alpha, gamma=args.focal_gamma)
    else:
        crit = nn.CrossEntropyLoss(weight=alpha, label_smoothing=args.label_smoothing)

    npos = int((tr['y'] == 1).sum()); nneg = int((tr['y'] == 0).sum())
    print(f"stage={args.stage} init={init}", flush=True)
    print(f"train candidates={len(tr['y'])} pos={npos} neg={nneg} | "
          f"sample/epoch≈{npos * (1 + args.neg_per_pos)} batch={args.batch} epochs={args.epochs}", flush=True)
    print(f"lr_head={args.lr_head} lr_backbone={args.lr_backbone} neg_per_pos={args.neg_per_pos} "
          f"workers={args.workers} loss={args.loss} gamma={args.focal_gamma} clip={args.clip_grad} "
          f"ls={args.label_smoothing} strong_aug={args.strong_aug} input={args.input_size} "
          f"in_chans={args.in_chans} gem={args.use_gem} cbam={args.use_cbam} od_gate={args.use_od_gate} "
          f"crop_center={args.crop_center} soft_mask={args.soft_mask} mask_sigma={args.mask_sigma} "
          f"mask_dilate={args.mask_dilate} mask_bg_sigma={args.mask_bg_sigma}", flush=True)
    if hard_pos is not None:
        print(f"hard positives={len(hard_pos)} x{args.hard_pos_mult}", flush=True)

    val_feats = None
    if args.stage == 1 and not args.no_val:
        t = time.time()
        print("预计算 val 特征（backbone 冻结，只需一次）...", flush=True)
        val_feats = compute_features(model, va, val_index, val_root, device, 256, args.workers, args.input_size, args.in_chans, str(Path(args.cellprob_dir) / 'val'),
                                     mask_dir=str(val_mask_dir), crop_center=args.crop_center,
                                     soft_mask=args.soft_mask, mask_sigma=args.mask_sigma,
                                         mask_dilate=args.mask_dilate, mask_bg_sigma=args.mask_bg_sigma,
                                         mask_channel=args.mask_channel)
        print(f"val 特征完成: {val_feats.shape} ({time.time()-t:.1f}s)", flush=True)

    best_f1 = -1.0
    bad = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        idx = make_epoch_index(tr['y'], args.neg_per_pos, rng, hard_pos, args.hard_pos_mult)
        ds = CandidateDataset('train', tr, idx, train_root, augment=True, strong=args.strong_aug, crop_size=args.input_size, in_chans=args.in_chans, cellprob_dir=str(Path(args.cellprob_dir) / 'train'),
                              mask_dir=str(train_mask_dir), crop_center=args.crop_center,
                              soft_mask=args.soft_mask, mask_sigma=args.mask_sigma,
                              mask_dilate=args.mask_dilate, mask_bg_sigma=args.mask_bg_sigma,
                                     mask_channel=args.mask_channel)
        dl = loader(ds, args.batch, args.workers, shuffle=False)
        model.train()
        tot = 0.0; nb = 0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model.predict_only(xb)
            loss = crit(logits, yb)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        msg = f"[ep {ep}/{args.epochs}] loss={tot/max(nb,1):.4f} time={time.time()-t0:.1f}s"
        if (not args.no_val) and (ep % args.eval_every == 0 or ep == args.epochs):
            te = time.time()
            if args.stage == 1:
                feats = val_feats
            else:
                feats = compute_features(model, va, val_index, val_root, device, 256, args.workers, args.input_size, args.in_chans, str(Path(args.cellprob_dir) / 'val'),
                                         mask_dir=str(val_mask_dir), crop_center=args.crop_center,
                                         soft_mask=args.soft_mask, mask_sigma=args.mask_sigma,
                              mask_dilate=args.mask_dilate, mask_bg_sigma=args.mask_bg_sigma,
                              mask_channel=args.mask_channel)
            best, fixed = eval_from_features(model, feats, va, val_index, gt_val, device)
            msg += (f" | val e2e F1={best['f1']:.4f} @thr={best['thr']:.2f} "
                    f"(P={best['p']:.3f} R={best['r']:.3f} TP={best['tp']} FP={best['fp']} FN={best['fn']})"
                    f" | @0.5 F1={fixed['f1']:.4f} | eval={time.time()-te:.1f}s")
            if best['f1'] > best_f1:
                best_f1 = best['f1']
                bad = 0
                torch.save(model.state_dict(), out / f"best_s{args.stage}{args.tag}.pth")
                (out / f"best_epoch_s{args.stage}{args.tag}.txt").write_text(str(ep))
                msg += " *saved*"
            else:
                bad += 1
                if bad >= args.patience:
                    print(msg + f" | early-stop (no improve {bad} evals)", flush=True)
                    break
        print(msg, flush=True)
        torch.save(model.state_dict(), out / f"last_s{args.stage}{args.tag}.pth")
    print(f"done. best val e2e F1={best_f1:.4f}" if not args.no_val else "done (no-val)", flush=True)


if __name__ == "__main__":
    main()
