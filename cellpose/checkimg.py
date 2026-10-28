import tifffile
import matplotlib.pyplot as plt
import imageio.v3 as iio   # 或使用 from PIL import Image
'''检查切割的细胞'''
img_path = "/home/hyw/dataset/chendawei/ICPR2014/mitos_atypia_2014_training_aperio/A03/frames/x40/A03_01Aa.tiff"
mask_path = "/home/hyw/chendawei/cellpose-4.2.1.1/cellpose/segimg/_2021-07-19_03_05_16_No_0_x_27092.0_y_25008_masks.tif"

# 使用 imageio 读取 PNG（自动返回 numpy 数组）
img = iio.imread(img_path)
# mask 依然是 tifffile 读取
mask = tifffile.imread(mask_path)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
# 如果 img 是彩色图 (H,W,3)，imshow 会自动显示彩色；若希望显示灰度，可以指定 cmap='gray'
# 但如果彩色图用 cmap='gray'，会警告或显示异常，最好判断一下通道数
if img.ndim == 2:
    plt.imshow(img, cmap="gray")
else:
    plt.imshow(img)  # 彩色显示

plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="nipy_spectral")
plt.title(f"Cellpose-SAM: {mask.max()} cells")
plt.axis("off")

plt.tight_layout()
plt.show()