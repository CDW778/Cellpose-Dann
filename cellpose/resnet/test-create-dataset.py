#!/usr/bin/env python3
"""
测试脚本：验证完整提取流程在单张图片上的表现
- 使用与完整脚本完全相同的参数和逻辑
- 输出详细统计信息（正样本数、困难负样本数、可用细胞数、补齐数等）
- 将生成的切片保存到 test_output 目录，方便人工检查
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
from cellpose import models, io, core
from tqdm import tqdm
import shutil

# ===================== 测试配置 =====================
# 请修改以下路径为您的实际路径
TEST_IMAGE = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train-val-test/test-original-img/A03_02Bb.tiff"
MITOSIS_CSV = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/mitosis_csv/A03_02Bb_mitosis.csv"
NOT_MITOSIS_CSV = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/not_mitosis_csv/A03_02Bb_not_mitosis.csv"

# Cellpose 参数（与完整脚本一致）
CELLPOSE_WEIGHT = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -1.0
TILE_NORM_BLOCKSIZE = 0

# 切片参数
PATCH_SIZE = 96
HALF_SIZE = PATCH_SIZE // 2
MIN_DIST_FROM_POS = 92   # 其他细胞与正样本的最小距离

# 输出目录（测试产生的切片将保存于此）
OUTPUT_DIR = "./test_output"
# ===================================================

def parse_icpr_csv(csv_path):
    coords = []
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return coords
    try:
        df = pd.read_csv(csv_path, header=None)
        if df.empty:
            return coords
        for _, row in df.iterrows():
            if len(row) >= 2:
                x, y = float(row[0]), float(row[1])
                coords.append((y, x))
    except Exception as e:
        print(f"读取CSV失败: {e}")
    return coords

def crop_patch_with_reflect(img, row_cen, col_cen):
    h, w = img.shape[:2]
    r, c = int(round(row_cen)), int(round(col_cen))
    r_min, r_max = r - HALF_SIZE, r + HALF_SIZE
    c_min, c_max = c - HALF_SIZE, c + HALF_SIZE
    if r_min < 0 or r_max > h or c_min < 0 or c_max > w:
        pad_top = max(0, -r_min)
        pad_bottom = max(0, r_max - h)
        pad_left = max(0, -c_min)
        pad_right = max(0, c_max - w)
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_REFLECT)
        return padded[r_min + pad_top:r_max + pad_top,
                      c_min + pad_left:c_max + pad_left]
    else:
        return img[r_min:r_max, c_min:c_max]

def load_cellpose_model(weight_path):
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")
    return models.CellposeModel(gpu=use_gpu, pretrained_model=weight_path)

def segment_single(model, img_path, flow_th, prob_th, tile_norm):
    """单张图像分割"""
    img = io.imread(img_path)
    mask, _, _ = model.eval(img,
                            batch_size=1,
                            flow_threshold=flow_th,
                            cellprob_threshold=prob_th,
                            normalize={"tile_norm_blocksize": tile_norm})
    return mask

def main():
    random.seed(42)
    np.random.seed(42)

    # 清理并创建输出目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print("加载 Cellpose 模型...")
    model = load_cellpose_model(CELLPOSE_WEIGHT)

    # 2. 读取原图
    img_bgr = cv2.imread(TEST_IMAGE)
    if img_bgr is None:
        print(f"无法读取图片: {TEST_IMAGE}")
        return
    h, w = img_bgr.shape[:2]
    base = os.path.splitext(os.path.basename(TEST_IMAGE))[0]
    print(f"图片: {base}, 尺寸: {h}x{w}")

    # 3. 分割
    print("运行 Cellpose 分割...")
    mask = segment_single(model, TEST_IMAGE, FLOW_THRESHOLD, CELLPROB_THRESHOLD, TILE_NORM_BLOCKSIZE)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    print(f"分割完成，唯一标签数: {len(np.unique(mask))}")

    # 4. 读取坐标
    pos_coords = parse_icpr_csv(MITOSIS_CSV)
    hard_neg_coords = parse_icpr_csv(NOT_MITOSIS_CSV)
    print(f"正样本点数量: {len(pos_coords)}")
    print(f"困难负样本点数量: {len(hard_neg_coords)}")

    # ========== 模拟完整流程 ==========
    # 正样本提取
    N = 0
    pos_mask_ids = set()
    error_count = 0
    pos_patches = []

    for idx, (r_gt, c_gt) in enumerate(pos_coords):
        r_int = int(np.clip(round(r_gt), 0, h-1))
        c_int = int(np.clip(round(c_gt), 0, w-1))
        mask_id = mask[r_int, c_int]

        if mask_id > 0:
            pos_mask_ids.add(mask_id)
            pixels = np.argwhere(mask == mask_id)
            r_cen, c_cen = pixels.mean(axis=0)
        else:
            r_cen, c_cen = r_gt, c_gt
            error_count += 1
            print(f"  [漏检] 正样本点 #{idx} (Y={r_gt}, X={c_gt}) 在Mask中为0")

        patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
        pos_patches.append(patch)
        N += 1

    print(f"正样本数 N = {N}, 漏检点数 = {error_count}")

    # 困难负样本提取
    K = 0
    hard_patches = []
    for idx, (r_gt, c_gt) in enumerate(hard_neg_coords):
        r_int = int(np.clip(round(r_gt), 0, h-1))
        c_int = int(np.clip(round(c_gt), 0, w-1))
        mask_id = mask[r_int, c_int]
        if mask_id > 0 and mask_id not in pos_mask_ids:
            pixels = np.argwhere(mask == mask_id)
            r_cen, c_cen = pixels.mean(axis=0)
            patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
            hard_patches.append(patch)
            K += 1

    print(f"困难负样本数 K = {K} (有效，未与正样本共用细胞)")

    target_neg = 3 * N
    print(f"目标负样本总数 (3N) = {target_neg}")

    # 策略判断
    if K >= target_neg:
        print("策略一：困难负样本足够，全量保留")
        final_neg = K
        cell_patches = []
    else:
        print("策略二：困难负样本不足，从其他细胞补齐")
        gap = target_neg - K
        print(f"  需要补齐 {gap} 个")

        # 收集其他细胞
        all_cell_ids = set(np.unique(mask)) - {0} - pos_mask_ids
        candidate_cells = []
        for cid in all_cell_ids:
            pixels = np.argwhere(mask == cid)
            if len(pixels) == 0:
                continue
            r_cen, c_cen = pixels.mean(axis=0)
            candidate_cells.append((r_cen, c_cen, cid))
        print(f"  候选其他细胞总数: {len(candidate_cells)}")

        # 距离筛选
        pos_array = np.array(pos_coords) if pos_coords else np.empty((0,2))
        valid_cells = []
        for (r_cen, c_cen, cid) in candidate_cells:
            if len(pos_array) > 0:
                dists = np.sqrt((pos_array[:,0] - r_cen)**2 + (pos_array[:,1] - c_cen)**2)
                min_dist = np.min(dists)
                if min_dist < MIN_DIST_FROM_POS:
                    continue
            valid_cells.append((r_cen, c_cen, cid))
        print(f"  距离正样本 >= {MIN_DIST_FROM_POS}px 的细胞数: {len(valid_cells)}")

        random.shuffle(valid_cells)
        chosen = valid_cells[:gap]
        print(f"  实际选取用于补齐的细胞数: {len(chosen)}")

        # 保存这些补齐细胞切片
        cell_patches = []
        for (r_cen, c_cen, _) in chosen:
            patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
            cell_patches.append(patch)

        final_neg = K + len(chosen)
        print(f"  最终负样本总数 = {final_neg}")

    # ========== 保存测试切片 ==========
    # 保存正样本
    pos_dir = os.path.join(OUTPUT_DIR, "pos")
    os.makedirs(pos_dir, exist_ok=True)
    for i, p in enumerate(pos_patches):
        cv2.imwrite(os.path.join(pos_dir, f"{base}_pos_{i}.png"), p)
    print(f"已保存正样本 {len(pos_patches)} 个到 {pos_dir}")

    # 保存困难负样本
    hard_dir = os.path.join(OUTPUT_DIR, "hard_neg")
    os.makedirs(hard_dir, exist_ok=True)
    for i, p in enumerate(hard_patches):
        cv2.imwrite(os.path.join(hard_dir, f"{base}_neg_hard_{i}.png"), p)
    print(f"已保存困难负样本 {len(hard_patches)} 个到 {hard_dir}")

    # 保存补齐细胞（如果有）
    if cell_patches:
        cell_dir = os.path.join(OUTPUT_DIR, "cell_neg")
        os.makedirs(cell_dir, exist_ok=True)
        for i, p in enumerate(cell_patches):
            cv2.imwrite(os.path.join(cell_dir, f"{base}_neg_cell_{i}.png"), p)
        print(f"已保存补齐细胞 {len(cell_patches)} 个到 {cell_dir}")

    # ========== 总结 ==========
    print("\n========== 测试总结 ==========")
    print(f"图片: {base}")
    print(f"正样本 N = {N}")
    print(f"困难负样本 K = {K}")
    print(f"目标负样本 3N = {target_neg}")
    print(f"策略: {'一' if K >= target_neg else '二'}")
    print(f"最终负样本总数 = {final_neg}")
    print(f"漏检正样本点数 = {error_count}")
    if K < target_neg:
        print(f"补齐细胞数 = {len(cell_patches)}")
    print(f"切片已保存至: {OUTPUT_DIR}")
    print("===============================")

if __name__ == "__main__":
    main()