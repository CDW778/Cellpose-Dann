#!/usr/bin/env python3
"""
数据集提取脚本（使用预先划分好的 train/val/test 大图目录）
- 不再随机划分，直接读取指定目录下的图片
- 策略一：困难负样本足够则全量保存
- 策略二：困难负样本不足，仅从其他细胞补齐（距离正样本 ≥ 92px），不够则不补
- 使用指定的 Cellpose 推理参数
"""

import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from cellpose import models, io, core
import warnings

warnings.filterwarnings("ignore")

# ===================== 用户配置 =====================
CONFIG = {
    # 已划分好的大图存放路径（分别对应训练、验证、测试集）
    "TRAIN_IMAGE_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train-val-test/train-original-img",   # 请修改为实际路径
    "VAL_IMAGE_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train-val-test/val-original-img",       # 请修改为实际路径
    "TEST_IMAGE_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train-val-test/test-original-img",     # 请修改为实际路径

    # CSV 文件目录（与之前相同）
    "MITOSIS_CSV_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/mitosis_csv",
    "NOT_MITOSIS_CSV_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/not_mitosis_csv",

    # 输出根目录（将在其中创建 train/val/test 子文件夹）
    "OUTPUT_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014",

    # Cellpose 模型及推理参数
    "CELLPOSE_WEIGHT": "/home/hyw/chendawei/cellpose-weight/cpsam_v2",
    "FLOW_THRESHOLD": 1.0,
    "CELLPROB_THRESHOLD": -1.0,
    "TILE_NORM_BLOCKSIZE": 0,
}

PATCH_SIZE = 96
HALF_SIZE = PATCH_SIZE // 2
MIN_DIST_FROM_POS = 50   # 其他细胞与正样本的最小距离（像素）
NEG_RATIO = 6            # 目标负正比（target_neg = NEG_RATIO * N）
DIST_UM = 8.0            # 末期邻近子核搜索半径（微米）
RES_APERIO = 0.2455
RES_HAM = (0.227299 + 0.227531) / 2.0
RANDOM_SEED = 42         # 用于随机选取细胞
BATCH_SIZE = 8           # Cellpose 批量分割大小，可据显存调整


def pixel_size(image_name):
    """按扫描仪返回像素分辨率（微米/像素）"""
    return RES_APERIO if image_name.startswith('A') else RES_HAM

# ====================================================

def parse_icpr_csv(csv_path):
    """解析 ICPR 坐标 CSV，返回 (Y, X) 列表"""
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
    except Exception:
        pass
    return coords

def crop_patch_with_reflect(img, row_cen, col_cen):
    """带反射填充的 96x96 切片"""
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

def segment_batch(model, img_paths, flow_th, prob_th, tile_norm):
    """批量分割，返回 mask 列表"""
    imgs = [io.imread(p) for p in img_paths]
    masks, _, _ = model.eval(imgs,
                             batch_size=len(imgs),
                             flow_threshold=flow_th,
                             cellprob_threshold=prob_th,
                             normalize={"tile_norm_blocksize": tile_norm})
    return masks

def process_images_for_split(image_dir, split_name, model_cp, log_f):
    """
    处理一个子集（train/val/test）的所有大图
    """
    # 获取该子集下的所有图片
    img_files = glob.glob(os.path.join(image_dir, "*.tiff"))
    if not img_files:
        img_files = glob.glob(os.path.join(image_dir, "*.tif"))
    if not img_files:
        print(f"警告：在 {image_dir} 中未找到图片，跳过 {split_name}")
        return

    pos_dir = os.path.join(CONFIG["OUTPUT_DIR"], split_name, "1_mitosis")
    neg_dir = os.path.join(CONFIG["OUTPUT_DIR"], split_name, "0_normal")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    error_dir = os.path.join(CONFIG["OUTPUT_DIR"], "cellpose_errors")
    os.makedirs(error_dir, exist_ok=True)

    # 批量处理
    for i in tqdm(range(0, len(img_files), BATCH_SIZE),
                  desc=f"处理 [{split_name.upper()}]"):
        batch_paths = img_files[i:i+BATCH_SIZE]
        masks = segment_batch(model_cp, batch_paths,
                              CONFIG["FLOW_THRESHOLD"],
                              CONFIG["CELLPROB_THRESHOLD"],
                              CONFIG["TILE_NORM_BLOCKSIZE"])

        for img_path, mask in zip(batch_paths, masks):
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            base = os.path.splitext(os.path.basename(img_path))[0]
            parts = base.split('_')
            core_prefix = parts[0] + '_' + parts[1] if len(parts) >= 2 else base

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"无法读取 {img_path}，跳过")
                continue
            h, w = img_bgr.shape[:2]

            # 查找 CSV
            pos_csv = os.path.join(CONFIG["MITOSIS_CSV_DIR"], f"{core_prefix}_mitosis.csv")
            neg_csv = os.path.join(CONFIG["NOT_MITOSIS_CSV_DIR"], f"{core_prefix}_not_mitosis.csv")
            pos_coords = parse_icpr_csv(pos_csv)
            hard_neg_coords = parse_icpr_csv(neg_csv)

            if not pos_coords and not hard_neg_coords:
                continue

            # ---------- 预计算所有 mask 质心（供正/负/末期使用） ----------
            mask_ids_all = np.unique(mask)
            mask_ids_all = mask_ids_all[mask_ids_all != 0]
            centroids = {}
            for lab in mask_ids_all:
                pixels = np.argwhere(mask == lab)
                centroids[lab] = pixels.mean(axis=0)  # (row, col)

            # ---------- 正样本提取 ----------
            N = 0
            pos_mask_ids = set()
            img_error_viz = None
            has_error = False
            thr_px = DIST_UM / pixel_size(base)

            for idx, (r_gt, c_gt) in enumerate(pos_coords):
                r_int = int(np.clip(round(r_gt), 0, h-1))
                c_int = int(np.clip(round(c_gt), 0, w-1))
                mask_id = mask[r_int, c_int]

                if mask_id > 0:
                    # GT 落在某 mask 内 → 用该 mask 质心
                    pos_mask_ids.add(mask_id)
                    r_cen, c_cen = centroids[mask_id]
                    patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
                    cv2.imwrite(os.path.join(pos_dir, f"{base}_pos_{idx}.png"), patch)
                    N += 1
                else:
                    # 末期缝隙（选项3）：找质心距 GT ≤ 8µm 的邻近子核，各切一个正样本
                    near = [lab for lab in mask_ids_all
                            if (centroids[lab][1] - c_gt) ** 2 + (centroids[lab][0] - r_gt) ** 2 <= thr_px ** 2]
                    if near:
                        k = 0
                        for lab in near:
                            if lab in pos_mask_ids:
                                continue
                            pos_mask_ids.add(lab)
                            r_cen, c_cen = centroids[lab]
                            patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
                            suffix = "" if k == 0 else f"_{k}"
                            cv2.imwrite(os.path.join(pos_dir, f"{base}_pos_{idx}{suffix}.png"), patch)
                            N += 1
                            k += 1
                        log_f.write(f"[末期] {base} 点 (Y:{r_gt}, X:{c_gt}) → {len(near)} 个邻近子核\n")
                    else:
                        # 真的没有邻近细胞 → 记录漏检
                        has_error = True
                        if img_error_viz is None:
                            img_error_viz = img_bgr.copy()
                        cv2.circle(img_error_viz, (int(round(c_gt)), int(round(r_gt))), 15, (0,0,255), 2)
                        log_f.write(f"[漏检] {base} 点 (Y:{r_gt}, X:{c_gt}) 漏检！无邻近细胞，跳过。\n")

            if has_error and img_error_viz is not None:
                cv2.imwrite(os.path.join(error_dir, f"{base}_errors.png"), img_error_viz)

            # ---------- 困难负样本提取 ----------
            K = 0
            all_hard_patches = []
            for idx, (r_gt, c_gt) in enumerate(hard_neg_coords):
                r_int = int(np.clip(round(r_gt), 0, h-1))
                c_int = int(np.clip(round(c_gt), 0, w-1))
                mask_id = mask[r_int, c_int]
                if mask_id > 0 and mask_id not in pos_mask_ids:
                    r_cen, c_cen = centroids[mask_id]
                    patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
                    all_hard_patches.append(patch)
                    K += 1

            target_neg = NEG_RATIO * N
            final_saved_neg = 0

            # ----- 策略一：困难负样本足够 -----
            if K >= target_neg:
                for idx, pth in enumerate(all_hard_patches):
                    cv2.imwrite(os.path.join(neg_dir, f"{base}_neg_hard_{idx}.png"), pth)
                final_saved_neg = K
                log_f.write(f"[策略一] {split_name}/{base} | N={N}, K={K} (K>={NEG_RATIO}N)，保留全部 {K} 个困难负样本\n")

            # ----- 策略二：困难负样本不足，仅从其他细胞补齐 -----
            else:
                # 保存已有困难负样本
                for idx, pth in enumerate(all_hard_patches):
                    cv2.imwrite(os.path.join(neg_dir, f"{base}_neg_hard_{idx}.png"), pth)
                final_saved_neg = K
                gap = target_neg - K

                # 收集其他细胞（排除正样本已用的）
                all_cell_ids = set(centroids.keys()) - pos_mask_ids
                pos_array = np.array(pos_coords) if pos_coords else np.empty((0, 2))
                valid_cells = []
                for cid in all_cell_ids:
                    r_cen, c_cen = centroids[cid]
                    if len(pos_array) > 0:
                        dists = np.sqrt((pos_array[:, 0] - r_cen) ** 2 + (pos_array[:, 1] - c_cen) ** 2)
                        min_dist = float(np.min(dists))
                        if min_dist < MIN_DIST_FROM_POS:
                            continue
                    else:
                        min_dist = 1e9
                    valid_cells.append((min_dist, r_cen, c_cen, cid))

                # 优先分裂邻近（距离升序）
                valid_cells.sort(key=lambda x: x[0])
                chosen_cells = valid_cells[:gap]   # 取前 gap 个，不够则全取
                cell_count = 0
                for (_, r_cen, c_cen, _cid) in chosen_cells:
                    patch = crop_patch_with_reflect(img_bgr, r_cen, c_cen)
                    cv2.imwrite(os.path.join(neg_dir, f"{base}_neg_cell_{cell_count}.png"), patch)
                    cell_count += 1
                    final_saved_neg += 1

                log_f.write(f"[策略二] {split_name}/{base} | N={N}, K={K} (K<{NEG_RATIO}N)，"
                            f"从其他细胞选取 {len(chosen_cells)} 个（目标 {gap} 个），"
                            f"最终负样本总数 {final_saved_neg} 个（可能少于3N）\n")

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 日志文件
    log_path = os.path.join(CONFIG["OUTPUT_DIR"], "slice_summary.log")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("================== 切片日志（使用预划分目录） ==================\n")

    # 加载 Cellpose 模型（只加载一次）
    model_cp = load_cellpose_model(CONFIG["CELLPOSE_WEIGHT"])

    # 定义子集及其对应的图片目录
    splits = {
        "train": CONFIG["TRAIN_IMAGE_DIR"],
        "val": CONFIG["VAL_IMAGE_DIR"],
        "test": CONFIG["TEST_IMAGE_DIR"],
    }

    for split_name, image_dir in splits.items():
        if not os.path.exists(image_dir):
            print(f"警告：目录不存在，跳过 {split_name}: {image_dir}")
            continue
        print(f"\n开始处理 {split_name} 集，图片目录: {image_dir}")
        process_images_for_split(image_dir, split_name, model_cp, log_f)

    log_f.write("================== 全部处理完成 ==================\n")
    log_f.close()
    print("\n所有切片生成完毕！")

if __name__ == "__main__":
    main()