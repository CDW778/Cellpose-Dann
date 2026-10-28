import os
import glob
'''计算细胞平均大小'''
# ================= 配置你的路径 =================
IMAGE_DIR = "/home/hyw/dataset/chendawei/top100/image"
LOCATION_TXT = "/home/hyw/dataset/chendawei/top100/train_location.txt"
OUTPUT_DIR = "/home/hyw/dataset/chendawei/train-classification"

# 1. 获取你选出的这 100 张大图的名字
all_png_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
local_imgs = {os.path.splitext(os.path.basename(f))[0].strip() for f in all_png_files}

# 2. 统计这 100 张大图在 TXT 里一共有多少个正样本标注
txt_pos_count = 0
with open(LOCATION_TXT, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines):
    parts = lines[i].split('\t')
    if len(parts) == 2 and parts[1].isdigit():
        img_name = parts[0].strip()
        num_objects = int(parts[1])

        # 如果这张大图在你的 100 张大图列表里，就累加它的正样本数量
        if img_name in local_imgs:
            txt_pos_count += num_objects

        i += 1
        # 跳过接下来的坐标行
        for _ in range(num_objects):
            if i >= len(lines): break
            if len(lines[i].split('\t')) == 4: i += 1  # 跳过bbox
            i += 1  # 跳过质心
    else:
        i += 1

# 3. 统计实际切出来的正样本小图数量 (包含 train, val, test 子目录)
actual_pos_images = (
        glob.glob(os.path.join(OUTPUT_DIR, "train/1_mitosis/*.png")) +
        glob.glob(os.path.join(OUTPUT_DIR, "val/1_mitosis/*.png")) +
        glob.glob(os.path.join(OUTPUT_DIR, "test/1_mitosis/*.png"))
)
actual_pos_count = len(actual_pos_images)

# ================= 打印核对结果 =================
print("\n" + "=" * 50)
print(f"【核对报告】")
print(f"1. 这 100 张大图在 TXT 文件中总共有： {txt_pos_count} 个正样本标注")
print(f"2. 最终文件夹里实际切出来的正样本有： {actual_pos_count} 张小图")
print("-" * 50)
if txt_pos_count == actual_pos_count:
    print(" 结果：【完全一致！】所有的正样本都完美切出来了。")
else:
    print(f" 结果：【不一致！】漏掉了 {txt_pos_count - actual_pos_count} 个正样本。")
print("=" * 50 + "\n")