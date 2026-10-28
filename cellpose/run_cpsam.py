#!/usr/bin/env python3
"""
Cellpose-SAM 批量分割脚本
用法：修改下方配置变量后直接运行
依赖：cellpose, numpy, matplotlib, pathlib, natsort, tqdm
安装：pip install cellpose numpy matplotlib natsort tqdm
"""

import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted

# ====================== 用户配置区域 ======================
# 图像目录（使用 Path 对象或字符串）
IMAGE_DIR = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train/train-original-img")          # 修改为你的图像文件夹

# 图像扩展名（如 ".tif", ".png", ".jpg"）
IMAGE_EXT = ".tiff"

# 通道选择（从0开始索引，支持多个通道，顺序无关）
# 示例：如果图像有3个通道，想使用第0和第2通道，则设置 channels=[0, 2]
# 如果图像为单通道灰度图，设置为 [0]
# 如果不想限制通道，可留空列表 [] 表示使用所有通道
CHANNELS = [0, 1, 2]          # 例如三通道荧光，可根据需要修改

# 分割阈值参数
FLOW_THRESHOLD = 1          # 流误差阈值，增大获得更多掩膜
CELLPROB_THRESHOLD = -0.8      # 细胞概率阈值，减小获得更多掩膜
TILE_NORM_BLOCKSIZE = 0       # 归一化块大小，0表示全图归一化

# 批处理模式：True 表示一次性加载所有图像后批量预测（适合小图），
# False 表示逐张处理（适合大图，省内存）
BATCH_MODE = False

# 模型选择：可选 "default" (Cellpose-SAM), "cpdino", "cpdino-vitb"
PRETRAINED_MODEL = "/home/hyw/chendawei/cellpose-weight/cpsam_v2"
# =========================================================

def main():
    # 1. 检查GPU
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")
    if not use_gpu:
        print("Warning: No GPU detected, running on CPU (may be slow).")

    # 2. 实例化模型
    if PRETRAINED_MODEL == "default":
        model = models.CellposeModel(gpu=use_gpu)
    else:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=PRETRAINED_MODEL)
    print(f"Model loaded: {PRETRAINED_MODEL}")

    # 3. 检查目录并获取文件列表
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {IMAGE_DIR}")
    files = natsorted([f for f in IMAGE_DIR.glob("*" + IMAGE_EXT)
                       if "_masks" not in f.name and "_flows" not in f.name])
    if not files:
        raise FileNotFoundError(f"No images with extension '{IMAGE_EXT}' found in {IMAGE_DIR}")
    print(f"Found {len(files)} images.")

    # 4. 确定输出格式（与输入扩展名一致，PNG 用 PNG，其余用 TIFF）
    masks_ext = ".png" if IMAGE_EXT.lower() == ".png" else ".tiff"

    # 5. 准备归一化参数
    normalize_kwargs = {"tile_norm_blocksize": TILE_NORM_BLOCKSIZE}

    # 6. 处理模式选择
    if BATCH_MODE:
        # 批量加载所有图像
        print("Loading all images...")
        imgs = [io.imread(f) for f in trange(len(files), desc="Loading")]
        # 处理通道选择
        if CHANNELS:
            # 假设所有图像通道数一致，取第一张的形状
            n_channels = imgs[0].shape[-1] if imgs[0].ndim == 3 else 1
            if any(c >= n_channels for c in CHANNELS):
                raise ValueError(f"Channel index out of range. Image has {n_channels} channels.")
            processed_imgs = []
            for img in imgs:
                if img.ndim == 2:
                    # 单通道
                    processed = np.expand_dims(img, axis=-1)
                else:
                    # 多通道，提取指定通道
                    processed = img[..., CHANNELS]
                processed_imgs.append(processed)
            imgs = processed_imgs
        else:
            # 使用所有通道，但确保为3D (H,W,C)
            for i, img in enumerate(imgs):
                if img.ndim == 2:
                    imgs[i] = np.expand_dims(img, axis=-1)

        print("Running Cellpose-SAM on batch...")
        masks, flows, styles = model.eval(imgs, batch_size=32,
                                          flow_threshold=FLOW_THRESHOLD,
                                          cellprob_threshold=CELLPROB_THRESHOLD,
                                          normalize=normalize_kwargs)

        print("Saving masks...")
        for i, f in enumerate(files):
            out_path = IMAGE_DIR / (f.stem + "_masks" + masks_ext)
            io.imsave(out_path, masks[i])

    else:
        # 逐张处理（节省内存）
        print("Processing images one by one...")
        for i, f in enumerate(files):
            print(f"Processing {f.name} ({i+1}/{len(files)})")
            img = io.imread(f)
            # 通道处理
            if img.ndim == 2:
                img_proc = np.expand_dims(img, axis=-1)
            else:
                if CHANNELS:
                    img_proc = img[..., CHANNELS]
                else:
                    img_proc = img
            # 预测
            masks, flows, styles = model.eval([img_proc], batch_size=1,
                                              flow_threshold=FLOW_THRESHOLD,
                                              cellprob_threshold=CELLPROB_THRESHOLD,
                                              normalize=normalize_kwargs)
            # 保存
            out_path = IMAGE_DIR / (f.stem + "_masks" + masks_ext)
            io.imsave(out_path, masks[0])

    print("All done!")


if __name__ == "__main__":
    main()