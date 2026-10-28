#!/usr/bin/env python3
"""用【原作者的 DANN 代码/规则】在 ICPR2012 test(04) 上评估。

要点：
- 模型直接 import 原仓库 Phase 2/final_model.py 的 DANNModel（stub 掉未安装的
  umap/seaborn；resnet50 用 weights=None 避免联网）。
- 预处理/transform/判定完全照原作：bbox 中心固定 224 原始 RGB 窗口、
  ToTensor+ImageNet Normalize、argmax 即 p_mitotic>=0.5。
- 候选 = test 全部 Cellpose mask（缓存 test_masks_cp3，已验证与现场等价）。

输出三层：
  1) 作者规则(argmax@0.5) patch 级 classification_report/AUC（全部候选）
  2) 对标作者平衡集：负样本下采样到 1:1 后同样报告
  3) 我们的判据：端到端 5µm 匹配 + 阈值扫描(oracle) 与固定 0.5
"""
import sys
import types
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from eval_2012_dann_live import res_um, load_gt_points, match, load_masks_from_cache
from dann_dataset import crop_patch
from gen_dataset_2012 import find_csv, label_bboxes, load_gt_regions, n_components
from coarse_filter_2012 import compute_features, passes

SRC_FINAL = ("/home/hyw/chendawei/cellpose-DANN-opencode/Biomedical-Data-Design-MIDOG-main/"
             "Phase 2/final_model.py")
DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
DANN_WEIGHT = "/home/hyw/chendawei/classification-weight/best_dann_model.pth"
NUM_DOMAIN_CLASSES = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
BATCH = 256
DIST_UM = 5.0


def load_author_module():
    for name in ['umap', 'seaborn']:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules['seaborn'].heatmap = lambda *a, **k: None
    sys.modules['umap'].UMAP = object
    spec = importlib.util.spec_from_file_location("author_final_model", SRC_FINAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', default=str(DST))
    ap.add_argument('--weight', default=DANN_WEIGHT)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--filter', default=None, help='几何粗筛阈值 json；只对存活 mask 打分')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--tta', action='store_true', help='翻转 TTA 概率平均')
    ap.add_argument('--fixed-thr', type=float, default=None, help='额外在指定阈值报告指标')
    args = ap.parse_args()

    dst = Path(args.dst)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 原作者模块 + 模型
    import torchvision
    _orig_resnet50 = torchvision.models.resnet50
    torchvision.models.resnet50 = lambda *a, **k: _orig_resnet50(weights=None)
    fm = load_author_module()
    model = fm.DANNModel(num_classes=2, num_domain_classes=NUM_DOMAIN_CLASSES, lambda_val=0.0)
    model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    model.eval().to(device)

    # 原作者 test transform：ToTensor + ImageNet Normalize
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print(f"device={device}; weight={args.weight}; model=author final_model.DANNModel", flush=True)

    thr_filter = None
    if args.filter:
        import json as _json
        thr_filter = _json.load(open(args.filter))['features_thresholds']
        print(f"应用几何粗筛: {args.filter}", flush=True)

    img_dir = dst / "train-val-test" / f"{args.split}-original-img"
    bmps = sorted(img_dir.glob("*.bmp"))

    recs = []          # (base, cx, cy, p, y)
    gt_by_hpf = {}
    total_gt = 0

    for hi, bmp in enumerate(bmps, 1):
        base = bmp.stem
        img_rgb = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
        px = res_um(base)
        csv_p = find_csv(base)
        gts = load_gt_points(csv_p) if csv_p else []
        gt_by_hpf[base] = (gts, DIST_UM / px)
        total_gt += len(gts)

        mp = dst / "cellpose_cache" / f"{args.split}_masks_cp3" / f"{base}_mask.npy"
        mask = np.load(mp) if mp.exists() else None
        if mask is None:
            print(f"[{hi}/{len(bmps)}] {base}: 无缓存 mask，跳过", flush=True); continue
        labels = [int(l) for l in np.unique(mask) if l != 0]
        cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
        bboxes = label_bboxes(mask)
        if thr_filter is not None:
            feats = compute_features(mask)
            keep = [l for l in labels if l in feats and passes(feats[l], thr_filter)]
            idx = [labels.index(l) for l in keep]
            labels = keep
            cents = [cents[i] for i in idx]

        # 正样本 = 普通(1 连通块) GT 区域的最大重叠 mask
        pos = set()
        if csv_p:
            for xs, ys in load_gt_regions(csv_p):
                if n_components(xs, ys) != 1:
                    continue
                labs, cnts = np.unique(mask[ys, xs], return_counts=True)
                nz = labs != 0
                if nz.sum() == 0:
                    continue
                pos.add(int(labs[np.argmax(np.where(nz, cnts, 0))]))

        patches, meta = [], []
        for lab, (cy, cx) in zip(labels, cents):
            y0, y1, x0, x1 = bboxes[lab]
            patches.append(crop_patch(img_rgb, (x0, y0, x1, y1), args.input_size))
            meta.append((float(cx), float(cy), 1 if lab in pos else 0))

        probs = []
        for i in range(0, len(patches), args.batch):
            chunk = patches[i:i + args.batch]
            tens = torch.stack([tf(Image.fromarray(p)) for p in chunk], 0).to(device)
            with torch.no_grad():
                if args.tta:
                    ps = [torch.softmax(model.predict_only(tens), 1)[:, 1]]
                    for dims in ([3], [2], [2, 3]):
                        ps.append(torch.softmax(model.predict_only(torch.flip(tens, dims)), 1)[:, 1])
                    p = torch.stack(ps, 0).mean(0)
                else:
                    p = torch.softmax(model.predict_only(tens), 1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())

        for (cx, cy, y), p in zip(meta, probs):
            recs.append((base, cx, cy, float(p), y))
        print(f"[{hi}/{len(bmps)}] {base}: masks={len(labels)} gt={len(gts)} pos={len(pos)}", flush=True)

    P = np.array([r[3] for r in recs]); Y = np.array([r[4] for r in recs])
    print(f"\n候选总数: {len(recs)}  正样本: {int(Y.sum())}  负样本: {int((Y == 0).sum())}  GT: {total_gt}")

    # ---------- 1) 作者规则 argmax@0.5（全部候选） ----------
    print("\n===== [1] 作者规则 argmax@0.5（全部 Cellpose 候选）=====")
    pred = (P >= 0.5).astype(int)
    print(classification_report(Y, pred, target_names=['Non-mitotic', 'Mitotic'], zero_division=0))
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(Y, pred))
    print(f"ROC-AUC (p_mitotic vs label) = {roc_auc_score(Y, P):.4f}")

    # ---------- 2) 对标作者平衡集：负样本下采样到 1:1 ----------
    print("\n===== [2] 平衡集(负样本下采样到 1:1) 作者规则 argmax@0.5 =====")
    rng = np.random.RandomState(args.seed)
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    neg_sel = rng.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)
    sel = np.concatenate([pos_idx, neg_sel])
    print(classification_report(Y[sel], (P[sel] >= 0.5).astype(int),
                                target_names=['Non-mitotic', 'Mitotic'], zero_division=0))
    print(f"ROC-AUC (balanced) = {roc_auc_score(Y[sel], P[sel]):.4f}")

    # ---------- 3) 我们的判据：端到端 5µm F1 ----------
    print("\n===== [3] 端到端 5µm F1（我们的判据）=====")
    by = {}
    for (base, cx, cy, p, y) in recs:
        by.setdefault(base, []).append((cx, cy, p))
    best = {'f1': -1}
    fixed05 = None
    fixed_custom = None
    for thr in np.arange(0.05, 0.9995, 0.005):
        tp = fp = fn = 0
        for base, rs in by.items():
            gts, thr_px = gt_by_hpf[base]
            dets = [(cx, cy, p) for (cx, cy, p) in rs if p >= thr]
            t, f, n = match(dets, gts, thr_px)
            tp += t; fp += f; fn += n
        pr = tp / (tp + fp) if (tp + fp) else 0
        rc = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
        if f1 > best['f1']:
            best = {'f1': f1, 'thr': thr, 'tp': tp, 'fp': fp, 'fn': fn, 'p': pr, 'r': rc}
        if abs(thr - 0.50) < 1e-9:
            fixed05 = {'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'p': pr, 'r': rc}
        if args.fixed_thr is not None and abs(thr - args.fixed_thr) < 1e-6:
            fixed_custom = {'f1': f1, 'thr': thr, 'tp': tp, 'fp': fp, 'fn': fn, 'p': pr, 'r': rc}
    print(f"oracle 最优: thr={best['thr']:.2f}  F1={best['f1']:.4f}  P={best['p']:.4f}  R={best['r']:.4f}  "
          f"TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    if fixed05:
        print(f"固定 0.5 : thr=0.50  F1={fixed05['f1']:.4f}  P={fixed05['p']:.4f}  R={fixed05['r']:.4f}  "
              f"TP={fixed05['tp']} FP={fixed05['fp']} FN={fixed05['fn']}")
    if fixed_custom:
        print(f"固定 {fixed_custom['thr']:.2f}: F1={fixed_custom['f1']:.4f}  P={fixed_custom['p']:.4f}  "
              f"R={fixed_custom['r']:.4f}  TP={fixed_custom['tp']} FP={fixed_custom['fp']} FN={fixed_custom['fn']}")


if __name__ == "__main__":
    main()
