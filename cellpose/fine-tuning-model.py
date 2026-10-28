import os
import glob
import cv2
import numpy as np
import pandas as pd
from cellpose import models
from tqdm import tqdm

# ========================== 用户配置区 ==========================
CONFIG = {
    "IMAGE_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/cellpose_errors/errors-original-img",
    "CSV_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/mitosis_csv",
    "ERROR_OUTPUT_DIR": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/cellpose_errors/fine-tuning-model-errors-img-7-6-11.19",
    "MODEL_PATH": "/home/hyw/chendawei/cellpose-weight/cpsam_v2",
    "LOG_FILE": "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/cellpose_errors/fine-tuning-model-errors-img-7-6-11.19/detection.log",
}

# 推理参数
EVAL_KWARGS = {
    "diameter":100,  # 默认 None，模型自动处理（不强制缩放）
    "flow_threshold": 1.0,  # 默认 0.4
    "cellprob_threshold": -0.8,  # 默认 0.0
    "niter": 600,  # 默认 None，自动根据 diameter 计算
    "min_size": 15,  # 默认 15
    "resample": True,  # 默认 True
    "augment": False,  # 默认 False
}

# ========================== 工具函数 ==========================
def parse_icpr_csv(csv_path):
    coords = []
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return coords
    try:
        df = pd.read_csv(csv_path, header=None)
        for _, row in df.iterrows():
            if len(row) >= 2:
                x, y = float(row[0]), float(row[1])
                coords.append((y, x))
    except Exception as e:
        print(f"读取 CSV 失败 {csv_path}: {e}")
    return coords

def draw_circles_on_image(img_bgr, points, radius=15, color=(0, 255, 0), thickness=2):
    img_copy = img_bgr.copy()
    for r, c in points:
        cv2.circle(img_copy, (int(round(c)), int(round(r))), radius, color, thickness)
    return img_copy

def generate_mask_visualization(mask, img_bgr=None, alpha=0.3):
    """
    生成整张 mask 的可视化图像
    返回: (color_mask, overlay_img)
        color_mask: 彩色标签图 (H,W,3) uint8，每个细胞随机颜色，背景黑色
        overlay_img: 在原图上叠加半透明彩色 mask 的图 (H,W,3) uint8
    """
    # 1. 生成彩色 mask（每个 label 一个随机颜色）
    labels = np.unique(mask)
    labels = labels[labels != 0]  # 去掉背景
    if len(labels) == 0:
        # 没有细胞，返回原图灰白
        color_mask = np.zeros_like(img_bgr) if img_bgr is not None else np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        overlay = img_bgr.copy() if img_bgr is not None else np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        return color_mask, overlay

    # 为每个标签随机生成颜色（固定种子保证同一图像内一致，但不同图像间无所谓）
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels)+1, 3), dtype=np.uint8)
    # 背景设为黑色
    colors[0] = [0, 0, 0]
    # 映射颜色
    color_mask = colors[mask]  # (H,W,3)

    # 2. 生成叠加图（如果提供了原图）
    if img_bgr is not None:
        overlay = img_bgr.copy()
        # 用半透明填充每个细胞区域
        for i, label in enumerate(labels):
            color = colors[label].tolist()
            # 创建该细胞的二值掩码
            cell_mask = (mask == label).astype(np.uint8) * 255
            # 找到轮廓并填充（也可直接使用掩码的布尔索引，但为了速度我们用轮廓填充）
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, -1)  # -1 填充
        # 混合原图和填充层
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
        # 再画上轮廓线（黑色细线）
        for i, label in enumerate(labels):
            cell_mask = (mask == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 0), 1)
    else:
        overlay = None

    return color_mask, overlay

# ========================== 主程序 ==========================
def main():
    os.makedirs(CONFIG["ERROR_OUTPUT_DIR"], exist_ok=True)
    log_path = CONFIG.get("LOG_FILE", None)
    log_f = open(log_path, "w", encoding="utf-8") if log_path else None

    image_paths = glob.glob(os.path.join(CONFIG["IMAGE_DIR"], "*.tiff"))
    image_paths += glob.glob(os.path.join(CONFIG["IMAGE_DIR"], "*.tif"))
    if not image_paths:
        print("错误：未找到任何 .tiff/.tif 图片，请检查 IMAGE_DIR 路径。")
        return

    print(f"找到 {len(image_paths)} 张图片，正在加载模型...")
    model = models.CellposeModel(gpu=True, pretrained_model=CONFIG["MODEL_PATH"])

    for img_path in tqdm(image_paths, desc="处理图片"):
        base = os.path.splitext(os.path.basename(img_path))[0]

        # 读取图像
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"无法读取图像 {img_path}，跳过")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 找 CSV
        csv_candidates = [
            os.path.join(CONFIG["CSV_DIR"], f"{base}_mitosis.csv"),
            os.path.join(CONFIG["CSV_DIR"], f"{base}.csv"),
            os.path.join(CONFIG["CSV_DIR"], f"{base.split('_')[0]}_{base.split('_')[1]}_mitosis.csv"),
        ]
        csv_path = None
        for cand in csv_candidates:
            if os.path.exists(cand):
                csv_path = cand
                break
        if csv_path is None:
            continue

        gt_coords = parse_icpr_csv(csv_path)
        if not gt_coords:
            continue

        # 模型推理
        masks, flows, styles = model.eval(img_rgb, **EVAL_KWARGS)

        # 检查漏检
        missed_points = []
        for r_gt, c_gt in gt_coords:
            r_int = int(np.clip(round(r_gt), 0, masks.shape[0]-1))
            c_int = int(np.clip(round(c_gt), 0, masks.shape[1]-1))
            if masks[r_int, c_int] == 0:
                missed_points.append((r_gt, c_gt))

        if missed_points:
            # 1. 保存画圈图（已有）
            img_with_circles = draw_circles_on_image(img_bgr, missed_points)
            out_circle = os.path.join(CONFIG["ERROR_OUTPUT_DIR"], f"{base}_errors.png")
            cv2.imwrite(out_circle, img_with_circles)

            # 2. 保存整张 mask 的可视化（新增）
            color_mask, overlay = generate_mask_visualization(masks, img_bgr, alpha=0.3)
            # 彩色标签图
            out_color = os.path.join(CONFIG["ERROR_OUTPUT_DIR"], f"{base}_mask_color.png")
            cv2.imwrite(out_color, color_mask)
            # 叠加图（原图+半透明mask）
            out_overlay = os.path.join(CONFIG["ERROR_OUTPUT_DIR"], f"{base}_mask_overlay.png")
            cv2.imwrite(out_overlay, overlay)

            if log_f:
                log_f.write(f"{base} 漏检 {len(missed_points)} 个点: {missed_points}\n")
            print(f"漏检！已保存: {out_circle} , {out_color} , {out_overlay}")

    if log_f:
        log_f.close()
    print("全部处理完成。")

if __name__ == "__main__":
    main()