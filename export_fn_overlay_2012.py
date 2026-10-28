#!/usr/bin/env python3
"""导出 FN 的真实推理输入（224 裁剪）+ 所有 GT/mask/质心 对比图。

每个 FN：左 = 模型实际看到的 224 原始裁剪；右 = 同一张图 + 标注。
标注：
  - 目标 GT：绿色（填充+轮廓+质心）
  - 同一 224 窗口内的其他 GT：青色（填充+轮廓+质心，标 gt id）
  - Cellpose mask：红色轮廓+淡红填充
  - mask 质心：蓝十字；mask bbox 中心（裁剪中心）：黄点
"""
import csv
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from gen_dataset_2012 import find_csv, load_gt_regions, n_components, label_bboxes

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT = Path("/home/hyw/chendawei/dann_finetune/fn_cases_v2")
SCORES = Path("/home/hyw/chendawei/dann_finetune/union_scores_test.npz")
THR = 0.97
PATCH = 224
HEAD = 26


def res_um(b):
    return 0.2456 if b.startswith("A") else (0.2273 + 0.22753) / 2.0


def crop_224(img, cx, cy):
    H, W = img.shape[:2]
    half = PATCH // 2
    l = max(0, cx - half); t = max(0, cy - half)
    r = l + PATCH; b = t + PATCH
    if r > W: r = W; l = r - PATCH
    if b > H: b = H; t = b - PATCH
    l = max(0, l); t = max(0, t)
    p = img[t:b, l:r]
    if p.shape[0] != PATCH or p.shape[1] != PATCH:
        p = cv2.resize(p, (PATCH, PATCH))
    return p, l, t


def panel(raw, ann, label):
    canvas = np.full((PATCH + HEAD, PATCH * 2 + 2, 3), 255, np.uint8)
    canvas[HEAD:, :PATCH] = raw
    canvas[HEAD:, PATCH + 2:] = ann
    cv2.putText(canvas, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _fill(ann, binm, color, alpha):
    if binm is not None and binm.any():
        ann[binm > 0] = (ann[binm > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def _contour(ann, binm, color):
    if binm is not None and binm.any():
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, cnts, -1, color, 1)


def make_ann(raw, mask_bin, target_bin, target_c, others, mask_c, bbox_c, gt_out, text):
    ann = raw.copy()
    # 其他 GT：青色
    for gid, gbin, gc in others:
        _fill(ann, gbin, (255, 255, 0), 0.30)
    # 目标 GT：绿色
    _fill(ann, target_bin, (0, 200, 0), 0.45)
    # mask：淡红
    _fill(ann, mask_bin, (0, 0, 200), 0.25)
    # 轮廓
    for gid, gbin, gc in others:
        _contour(ann, gbin, (255, 255, 0))
    _contour(ann, target_bin, (0, 255, 0))
    _contour(ann, mask_bin, (0, 0, 255))
    # 其他 GT 质心 + id
    for gid, gbin, gc in others:
        if gc is not None:
            cv2.drawMarker(ann, gc, (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(ann, f"gt{gid}", (gc[0] + 3, gc[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    if bbox_c is not None:
        cv2.circle(ann, bbox_c, 3, (0, 220, 255), -1)
    if mask_c is not None:
        cv2.drawMarker(ann, mask_c, (255, 0, 0), cv2.MARKER_CROSS, 12, 1)
    if target_c is not None:
        cv2.drawMarker(ann, target_c, (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
    for i, line in enumerate(text):
        cv2.putText(ann, line, (3, 14 + i * 13), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    if gt_out:
        cv2.putText(ann, f"target GT outside {gt_out}%", (3, PATCH - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return ann


def contact_sheet(items, path, cols):
    tile_w = PATCH * 2 + 2; tile_h = PATCH + HEAD
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (255, 255, 255))
    for k, p in enumerate(items):
        r, c = divmod(k, cols)
        canvas.paste(Image.open(p), (c * tile_w, r * tile_h))
    canvas.save(path)


def rgb_to_crop_xy(xs, ys, l, t):
    xx = xs - l; yy = ys - t
    ok = (xx >= 0) & (xx < PATCH) & (yy >= 0) & (yy < PATCH)
    return xx[ok], yy[ok], ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SCORES, allow_pickle=True)
    by = {}
    for b in np.unique(d['base']):
        m = d['base'] == b
        by[str(b)] = {int(l): (float(cx), float(cy), float(pd), float(pc))
                      for cx, cy, pd, pc, l in zip(d['cx'][m], d['cy'][m], d['pD'][m], d['pC'][m], d['label'][m])}

    rows = []
    for base in sorted(by):
        img = cv2.imread(str(DST / "train-val-test" / "test-original-img" / f"{base}.bmp"))
        mask = np.load(DST / "cellpose_cache" / "test_masks_cp3" / f"{base}_mask.npy")
        bb = label_bboxes(mask)
        regs = load_gt_regions(find_csv(base))
        thr_px = 5.0 / res_um(base)
        det = by[base]

        # greedy match at 0.97 -> FN
        didx = [l for l, v in det.items() if v[2] >= THR]
        used = set(); matched = set()
        for gi, (xs, ys) in enumerate(regs):
            gx, gy = xs.mean(), ys.mean()
            for dist, l in sorted([(np.hypot(det[l][0] - gx, det[l][1] - gy), l) for l in didx]):
                if dist > thr_px: break
                if l in used: continue
                used.add(l); matched.add(gi); break

        for gi, (xs, ys) in enumerate(regs):
            if gi in matched:
                continue
            nc = n_components(xs, ys)
            gx, gy = int(round(xs.mean())), int(round(ys.mean()))
            gta = len(xs)
            # 选最大 IoU 的 mask
            labs = [int(l) for l in np.unique(mask[ys, xs]) if l != 0]
            best = (0.0, -1)
            for l in labs:
                inter = int(np.sum(mask[ys, xs] == l))
                ma = int(np.sum(mask == l)); iou = inter / (gta + ma - inter)
                if iou > best[0]:
                    best = (iou, l)
            iou, lab = best
            if lab == -1:
                dd = [(np.hypot(det[l][0] - gx, det[l][1] - gy), l) for l in det]
                dd = [x for x in dd if x[0] <= thr_px]
                lab = min(dd)[1] if dd else -1
            if lab == -1:
                continue
            y0, y1, x0, x1 = bb[lab]
            bcx, bcy = (x0 + x1) // 2, (y0 + y1) // 2
            raw, l, t = crop_224(img, bcx, bcy)

            # 目标 GT 在窗口内比例
            _, _, ok_t = rgb_to_crop_xy(xs, ys, l, t)
            inside = int(ok_t.sum())
            gt_out = round(100 * (1 - inside / gta), 1)

            # 窗口内所有 GT
            target_bin = None; target_c = None; others = []; ids_in = []
            for gj, (gxs, gys) in enumerate(regs):
                xx, yy, ok = rgb_to_crop_xy(gxs, gys, l, t)
                if len(xx) == 0:
                    continue
                ids_in.append(gj)
                gbin = np.zeros((PATCH, PATCH), np.uint8)
                gbin[yy, xx] = 1
                cx = int(round(gxs[ok].mean())) - l
                cy = int(round(gys[ok].mean())) - t
                gc = (cx, cy) if (0 <= cx < PATCH and 0 <= cy < PATCH) else None
                if gj == gi:
                    target_bin = gbin; target_c = gc
                else:
                    others.append((gj, gbin, gc))

            # mask
            mfull = (mask == lab).astype(np.uint8)
            mask_bin = np.zeros((PATCH, PATCH), np.uint8)
            mc = mfull[t:t + PATCH, l:l + PATCH]
            mask_bin[:mc.shape[0], :mc.shape[1]] = mc
            yx = np.argwhere(mfull > 0).mean(0)
            mask_c = (int(round(yx[1] - l)), int(round(yx[0] - t)))
            bbox_c = (bcx - l, bcy - t)
            def inside_pt(p):
                return 0 <= p[0] < PATCH and 0 <= p[1] < PATCH
            mask_c = mask_c if inside_pt(mask_c) else None
            bbox_c = bbox_c if inside_pt(bbox_c) else None

            pD = det[lab][2] if lab in det else -1.0
            pC = det[lab][3] if lab in det else -1.0
            cdist = np.hypot(det[lab][0] - gx, det[lab][1] - gy) * res_um(base) if lab in det else -1.0
            gts_str = ",".join(f"gt{i}" for i in ids_in)
            text = [f"{base} gt{gi} nc{nc} IoU={iou:.2f} pD={pD:.2f}",
                    f"GTa={gta} Ma={int(mfull.sum())} cdist={cdist:.1f}um",
                    f"GTs in crop: {gts_str}"]
            ann = make_ann(raw, mask_bin, target_bin, target_c, others, mask_c, bbox_c, gt_out, text)
            tile = panel(raw, ann, f"{base} gt{gi} nc{nc} IoU{iou:.2f} p{pD:.2f}")
            fn = f"{base}_gt{gi}_nc{nc}_iou{iou:.2f}_p{pD:.2f}.png"
            cv2.imwrite(str(OUT / fn), tile)
            extra = [g for g in ids_in if g != gi]
            rows.append(dict(base=base, gt=gi, ncomp=nc, gt_area=gta, mask_area=int(mfull.sum()),
                             iou=round(iou, 3), pD=round(pD, 4), pC=round(pC, 4),
                             centroid_dist_um=round(float(cdist), 2), gt_outside_pct=gt_out,
                             label=lab, n_gt_in_crop=len(ids_in),
                             extra_gt_in_crop=";".join(str(g) for g in extra), file=fn))

    with open(OUT / "fn_overlay_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    normal = [str(OUT / r['file']) for r in rows if r['ncomp'] == 1]
    telo = [str(OUT / r['file']) for r in rows if r['ncomp'] >= 2]
    allf = [str(OUT / r['file']) for r in rows]
    contact_sheet(normal, OUT / "contact_sheet_normal21.png", 4)
    contact_sheet(telo, OUT / "contact_sheet_telophase4.png", 2)
    contact_sheet(allf, OUT / "contact_sheet_all25.png", 4)
    print(f"FN total={len(rows)} normal={len(normal)} telophase={len(telo)}")
    print(f"输出目录: {OUT}")
    for p in [OUT / "contact_sheet_normal21.png", OUT / "contact_sheet_telophase4.png",
              OUT / "contact_sheet_all25.png", OUT / "fn_overlay_cases.csv"]:
        print(f"  {p}")


if __name__ == '__main__':
    main()
