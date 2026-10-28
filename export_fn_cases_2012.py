#!/usr/bin/env python3
"""导出 seed0(thr=0.97) 的 FN 细胞裁剪图 + 标注 + 拼图 + CSV。"""
import csv
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw

from gen_dataset_2012 import find_csv, load_gt_regions, n_components, label_bboxes

DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
OUT = Path("/home/hyw/chendawei/dann_finetune/fn_cases")
SCORES = Path("/home/hyw/chendawei/dann_finetune/union_scores_test.npz")
THR = 0.97


def res_um(b):
    return 0.2456 if b.startswith("A") else (0.2273 + 0.22753) / 2.0


def crop_window(img, cx, cy, size):
    H, W = img.shape[:2]
    half = size // 2
    l = max(0, cx - half); t = max(0, cy - half)
    r = l + size; b = t + size
    if r > W: r = W; l = r - size
    if b > H: b = H; t = b - size
    l = max(0, l); t = max(0, t)
    p = img[t:b, l:r]
    if p.shape[0] != size or p.shape[1] != size:
        p = cv2.resize(p, (size, size))
    return p, l, t


def contact_sheet(items, path, cols, title):
    tile = 224; head = 26
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * (tile + head)), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, (img_path, label) in enumerate(items):
        r, c = divmod(k, cols)
        x = c * tile; y = r * (tile + head)
        d.text((x + 4, y + 6), label, fill=(0, 0, 0))
        im = Image.open(img_path)
        canvas.paste(im, (x, y + head))
    canvas.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SCORES, allow_pickle=True)
    by = {}
    for b in np.unique(d['base']):
        m = d['base'] == b
        by[str(b)] = dict(cx=d['cx'][m], cy=d['cy'][m], pd=d['pD'][m], lab=d['label'][m])

    rows = []
    for base in sorted(by):
        img_bgr = cv2.imread(str(DST / "train-val-test" / "test-original-img" / f"{base}.bmp"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = np.load(DST / "cellpose_cache" / "test_masks_cp3" / f"{base}_mask.npy")
        bb = label_bboxes(mask)
        regs = load_gt_regions(find_csv(base))
        thr_px = 5.0 / res_um(base)
        det = by[base]
        didx = np.where(det['pd'] >= THR)[0]
        used = set(); matched = set()
        for gi, (xs, ys) in enumerate(regs):
            if len(didx) == 0: break
            gx, gy = xs.mean(), ys.mean()
            dd = np.sqrt((det['cx'][didx] - gx) ** 2 + (det['cy'][didx] - gy) ** 2)
            for k in np.argsort(dd):
                if dd[k] > thr_px: break
                j = int(didx[k])
                if j in used: continue
                used.add(j); matched.add(gi); break
        for gi, (xs, ys) in enumerate(regs):
            if gi in matched: continue
            gx, gy = int(round(xs.mean())), int(round(ys.mean()))
            nc = n_components(xs, ys)
            dd = np.sqrt((det['cx'] - gx) ** 2 + (det['cy'] - gy) ** 2)
            near = np.where(dd <= thr_px)[0]
            if len(near):
                j = int(near[np.argmin(dd[near])])
                lab = int(det['lab'][j]); p = float(det['pd'][j])
                dist_um = float(dd[j] * res_um(base))
                y0, y1, x0, x1 = bb[lab]; area = (y1 - y0) * (x1 - x0)
            else:
                lab = -1; p = -1.0; dist_um = -1.0; area = -1
            rgb = img_rgb[ys, xs].mean(0)

            p224, l, t = crop_window(img_rgb, gx, gy, 224)
            p112, _, _ = crop_window(img_rgb, gx, gy, 112)
            # annotate 224 (RGB -> BGR for cv2 draw)
            ann = cv2.cvtColor(p224.copy(), cv2.COLOR_RGB2BGR)
            if lab > 0:
                cnts, _ = cv2.findContours((mask == lab).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                shifted = [c - np.array([[l, t]]) for c in cnts]
                cv2.drawContours(ann, shifted, -1, (0, 0, 255), 1)
            cv2.drawMarker(ann, (gx - l, gy - t), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(ann, f"{base} gt{gi} ncomp{nc} p={p:.2f}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            fn = f"{base}_gt{gi}_nc{nc}_p{p:.2f}.png"
            cv2.imwrite(str(OUT / fn), ann)
            cv2.imwrite(str(OUT / ("112_" + fn)), cv2.cvtColor(p112, cv2.COLOR_RGB2BGR))

            rows.append(dict(base=base, gt=gi, ncomp=nc, area=area, p_near=round(p, 4),
                             dist_um=round(dist_um, 2), R=round(float(rgb[0]), 1),
                             G=round(float(rgb[1]), 1), B=round(float(rgb[2]), 1),
                             nearest_label=lab, file=fn))

    with open(OUT / "fn_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    normal = [(str(OUT / r['file']), f"{r['base']} gt{r['gt']} p={r['p_near']:.2f}") for r in rows if r['ncomp'] == 1]
    telo = [(str(OUT / r['file']), f"{r['base']} gt{r['gt']} TELO p={r['p_near']:.2f}") for r in rows if r['ncomp'] >= 2]
    contact_sheet(normal, OUT / "fn_contact_sheet_normal21.png", 7, "normal")
    contact_sheet(telo, OUT / "fn_contact_sheet_telophase4.png", 2, "telophase")
    contact_sheet([(str(OUT / r['file']), f"{r['base']} gt{r['gt']} nc{r['ncomp']}") for r in rows],
                  OUT / "fn_contact_sheet_all25.png", 5, "all")
    print(f"FN total={len(rows)} normal={len(normal)} telophase={len(telo)}")
    print(f"输出目录: {OUT}")
    print(f"  {OUT/'fn_contact_sheet_normal21.png'}")
    print(f"  {OUT/'fn_contact_sheet_telophase4.png'}")
    print(f"  {OUT/'fn_contact_sheet_all25.png'}")
    print(f"  {OUT/'fn_cases.csv'}")


if __name__ == '__main__':
    main()
