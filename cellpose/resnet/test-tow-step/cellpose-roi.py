#!/usr/bin/env python3
"""
提取 Cellpose 分割的细胞 ROI，按类别保存为文件夹，
用于独立测试分类器（tools/test.py）
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from cellpose import models, core, io
from scipy.spatial import KDTree

# ===================== 用户配置 =====================
IMAGE_DIR = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train-val-test/test-original-img")
IMAGE_EXT = ".tiff"

CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -1.0
TILE_NORM_BLOCKSIZE = 0

GT_DIR = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2014/mitosis_csv")

RES_APERIO = 0.2455
RES_HAMAMATSU_H = 0.227299
RES_HAMAMATSU_V = 0.227531
DIST_THRESHOLD_UM = 8.0   # 与 pipeline 一致

# 输出目录（保存 ROI 图片）
OUTPUT_ROOT = Path("/home/hyw/chendawei/cellpose-4.2.1.1/cellpose/segimg")

MIN_CELL_AREA = 10
CROP_SIZE = 96   # 固定正方形

USE_GPU = True

# ===== 新增：只处理前 N 张图片 =====
MAX_IMAGES = 10
# ===================================

def get_pixel_size(image_name):
    if image_name.startswith('A'):
        return RES_APERIO, RES_APERIO
    elif image_name.startswith('H'):
        return RES_HAMAMATSU_H, RES_HAMAMATSU_V
    else:
        return RES_APERIO, RES_APERIO

def load_cellpose_model():
    use_gpu = core.use_gpu() if USE_GPU else False
    print(f"GPU available: {use_gpu}")
    return models.CellposeModel(gpu=use_gpu, pretrained_model=CELLPOSE_WEIGHT)

def load_ground_truth(gt_dir, image_stem):
    gt_file = gt_dir / f"{image_stem}_mitosis.csv"
    if not gt_file.exists():
        return None
    points = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                points.append((x, y))
    return points

def extract_roi_from_mask(mask, img_bgr, min_area=MIN_CELL_AREA):
    h, w = img_bgr.shape[:2]
    cells = []
    labels = np.unique(mask)
    labels = labels[labels != 0]
    half = CROP_SIZE // 2

    for label in labels:
        ys, xs = np.where(mask == label)
        if len(ys) < min_area:
            continue
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        x1 = cx - half
        x2 = cx + half
        y1 = cy - half
        y2 = cy + half

        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)

        roi = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if pad_left or pad_right or pad_top or pad_bottom:
            roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_REFLECT)

        if roi.shape[0] != CROP_SIZE or roi.shape[1] != CROP_SIZE:
            roi = cv2.resize(roi, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)

        cells.append({
            'roi_bgr': roi,
            'cx': cx,
            'cy': cy
        })
    return cells

def main():
    print("Loading Cellpose...")
    model_cp = load_cellpose_model()

    # 创建输出目录（按类别）
    normal_dir = OUTPUT_ROOT / "0_normal"
    mitosis_dir = OUTPUT_ROOT / "1_mitosis"
    normal_dir.mkdir(parents=True, exist_ok=True)
    mitosis_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGE_DIR.glob("*" + IMAGE_EXT))
    if not image_files:
        raise FileNotFoundError(f"No images with extension {IMAGE_EXT} in {IMAGE_DIR}")

    print(f"Total images: {len(image_files)}")

    # 只处理有 GT 的图片
    valid_files = []
    for f in image_files:
        gts = load_ground_truth(GT_DIR, f.stem)
        if gts is not None:
            valid_files.append((f, gts))
    print(f"Images with GT: {len(valid_files)}")

    # ===== 限制只处理前 MAX_IMAGES 张 =====
    if len(valid_files) > MAX_IMAGES:
        valid_files = valid_files[:MAX_IMAGES]
        print(f"[限制] 只处理前 {MAX_IMAGES} 张图片")
    # =======================================

    total_cells = 0
    pos_count = 0
    neg_count = 0

    for img_path, gts in tqdm(valid_files, desc="Processing images"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Cannot read {img_path}, skipping.")
            continue

        # 分割
        masks, _, _ = model_cp.eval([img_bgr],
                                    batch_size=1,
                                    flow_threshold=FLOW_THRESHOLD,
                                    cellprob_threshold=CELLPROB_THRESHOLD,
                                    normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE})
        mask = masks[0]  # 单张

        # 提取 ROI
        cells = extract_roi_from_mask(mask, img_bgr)

        if not cells:
            continue

        # 距离阈值（像素）
        img_name = img_path.name
        h_res, v_res = get_pixel_size(img_name)
        pixel_size = (h_res + v_res) / 2.0 if img_name.startswith('H') else h_res
        threshold_px = DIST_THRESHOLD_UM / pixel_size

        # 构建 GT 点 KDTree
        if gts:
            tree = KDTree(gts)
        else:
            tree = None

        # 对每个细胞判断是否阳性
        for idx, cell in enumerate(cells):
            cx, cy = cell['cx'], cell['cy']
            is_mitosis = False
            if tree is not None:
                dist, _ = tree.query([cx, cy])
                if dist <= threshold_px:
                    is_mitosis = True

            # 保存图片
            base_name = img_path.stem
            filename = f"{base_name}_cell{idx:04d}.png"
            if is_mitosis:
                out_path = mitosis_dir / filename
                pos_count += 1
            else:
                out_path = normal_dir / filename
                neg_count += 1
            cv2.imwrite(str(out_path), cell['roi_bgr'])
            total_cells += 1

    print(f"\n✅ 提取完成！共保存 {total_cells} 个细胞 ROI")
    print(f"   阳性（有丝分裂）: {pos_count}")
    print(f"   阴性（正常）    : {neg_count}")
    print(f"输出目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
提取 Cellpose 分割的细胞 ROI，按类别保存为文件夹，
用于独立测试分类器（tools/test.py）
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from cellpose import models, core, io
from scipy.spatial import KDTree

# ===================== 用户配置 =====================
IMAGE_DIR = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train-val-test/test-original-img")
IMAGE_EXT = ".tiff"

CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -1.0
TILE_NORM_BLOCKSIZE = 0

GT_DIR = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2014/mitosis_csv")

RES_APERIO = 0.2455
RES_HAMAMATSU_H = 0.227299
RES_HAMAMATSU_V = 0.227531
DIST_THRESHOLD_UM = 8.0   # 与 pipeline 一致

# 输出目录（保存 ROI 图片）
OUTPUT_ROOT = Path("/home/hyw/dataset/chendawei/roi_extracted_for_test/test10")

MIN_CELL_AREA = 10
CROP_SIZE = 96   # 固定正方形

USE_GPU = True

# ===== 新增：只处理前 N 张图片 =====
MAX_IMAGES = 10
# ===================================

def get_pixel_size(image_name):
    if image_name.startswith('A'):
        return RES_APERIO, RES_APERIO
    elif image_name.startswith('H'):
        return RES_HAMAMATSU_H, RES_HAMAMATSU_V
    else:
        return RES_APERIO, RES_APERIO

def load_cellpose_model():
    use_gpu = core.use_gpu() if USE_GPU else False
    print(f"GPU available: {use_gpu}")
    return models.CellposeModel(gpu=use_gpu, pretrained_model=CELLPOSE_WEIGHT)

def load_ground_truth(gt_dir, image_stem):
    gt_file = gt_dir / f"{image_stem}_mitosis.csv"
    if not gt_file.exists():
        return None
    points = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                points.append((x, y))
    return points

def extract_roi_from_mask(mask, img_bgr, min_area=MIN_CELL_AREA):
    h, w = img_bgr.shape[:2]
    cells = []
    labels = np.unique(mask)
    labels = labels[labels != 0]
    half = CROP_SIZE // 2

    for label in labels:
        ys, xs = np.where(mask == label)
        if len(ys) < min_area:
            continue
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        x1 = cx - half
        x2 = cx + half
        y1 = cy - half
        y2 = cy + half

        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)

        roi = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if pad_left or pad_right or pad_top or pad_bottom:
            roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_REFLECT)

        if roi.shape[0] != CROP_SIZE or roi.shape[1] != CROP_SIZE:
            roi = cv2.resize(roi, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)

        cells.append({
            'roi_bgr': roi,
            'cx': cx,
            'cy': cy
        })
    return cells

def main():
    print("Loading Cellpose...")
    model_cp = load_cellpose_model()

    # 创建输出目录（按类别）
    normal_dir = OUTPUT_ROOT / "0_normal"
    mitosis_dir = OUTPUT_ROOT / "1_mitosis"
    normal_dir.mkdir(parents=True, exist_ok=True)
    mitosis_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(IMAGE_DIR.glob("*" + IMAGE_EXT))
    if not image_files:
        raise FileNotFoundError(f"No images with extension {IMAGE_EXT} in {IMAGE_DIR}")

    print(f"Total images: {len(image_files)}")

    # 只处理有 GT 的图片
    valid_files = []
    for f in image_files:
        gts = load_ground_truth(GT_DIR, f.stem)
        if gts is not None:
            valid_files.append((f, gts))
    print(f"Images with GT: {len(valid_files)}")

    # ===== 限制只处理前 MAX_IMAGES 张 =====
    if len(valid_files) > MAX_IMAGES:
        valid_files = valid_files[:MAX_IMAGES]
        print(f"[限制] 只处理前 {MAX_IMAGES} 张图片")
    # =======================================

    total_cells = 0
    pos_count = 0
    neg_count = 0

    for img_path, gts in tqdm(valid_files, desc="Processing images"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Cannot read {img_path}, skipping.")
            continue

        # 分割
        masks, _, _ = model_cp.eval([img_bgr],
                                    batch_size=1,
                                    flow_threshold=FLOW_THRESHOLD,
                                    cellprob_threshold=CELLPROB_THRESHOLD,
                                    normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE})
        mask = masks[0]  # 单张

        # 提取 ROI
        cells = extract_roi_from_mask(mask, img_bgr)

        if not cells:
            continue

        # 距离阈值（像素）
        img_name = img_path.name
        h_res, v_res = get_pixel_size(img_name)
        pixel_size = (h_res + v_res) / 2.0 if img_name.startswith('H') else h_res
        threshold_px = DIST_THRESHOLD_UM / pixel_size

        # 构建 GT 点 KDTree
        if gts:
            tree = KDTree(gts)
        else:
            tree = None

        # 对每个细胞判断是否阳性
        for idx, cell in enumerate(cells):
            cx, cy = cell['cx'], cell['cy']
            is_mitosis = False
            if tree is not None:
                dist, _ = tree.query([cx, cy])
                if dist <= threshold_px:
                    is_mitosis = True

            # 保存图片
            base_name = img_path.stem
            filename = f"{base_name}_cell{idx:04d}.png"
            if is_mitosis:
                out_path = mitosis_dir / filename
                pos_count += 1
            else:
                out_path = normal_dir / filename
                neg_count += 1
            cv2.imwrite(str(out_path), cell['roi_bgr'])
            total_cells += 1

    print(f"\n✅ 提取完成！共保存 {total_cells} 个细胞 ROI")
    print(f"   阳性（有丝分裂）: {pos_count}")
    print(f"   阴性（正常）    : {neg_count}")
    print(f"输出目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()