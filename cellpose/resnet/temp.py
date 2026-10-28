import os
import shutil

# ================= 请在这里修改您的路径 =================
SOURCE_DIR = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/original-img"     # 原始图片文件夹

# 三个前缀文档（每行一个前缀）
PREFIX_FILE1 = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train/train-original-img-ID.txt"
PREFIX_FILE2 = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/val/val-original-img-ID.txt"
PREFIX_FILE3 = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/test/test-original-img-ID.txt"

# 三个目标文件夹（复制到此处）
DEST_DIR1 = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train/train-original-img"
DEST_DIR2 ="/home/hyw/dataset/chendawei/train-classification/ICPR-2014/val/val-original-img"
DEST_DIR3 = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/test/test-original-img"
# ======================================================

def load_prefixes(filepath):
    """读取前缀文档，返回去重后的前缀集合（跳过空行）"""
    prefixes = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip()
                if p:
                    prefixes.add(p)
    except FileNotFoundError:
        print(f"错误：文件 '{filepath}' 不存在，请检查路径")
        exit(1)
    except Exception as e:
        print(f"读取文件 '{filepath}' 时发生错误：{e}")
        exit(1)
    return prefixes

def ensure_dir(directory):
    """确保目录存在，若不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录：{directory}")

def copy_matching_files(source_dir, dest_dirs, prefix_sets, dest_labels):
    """
    遍历源目录，将匹配前缀的文件复制到对应的目标目录。
    dest_dirs: 目标目录列表（与 prefix_sets 顺序对应）
    prefix_sets: 前缀集合列表
    dest_labels: 标签列表，仅用于输出信息
    """
    # 确保所有目标目录存在
    for d in dest_dirs:
        ensure_dir(d)

    # 遍历源目录中的文件
    files = os.listdir(source_dir)
    total_files = len(files)
    copied_counts = [0] * len(dest_dirs)

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(source_dir, filename)
        if not os.path.isfile(filepath):
            continue

        # 检查每个文档的前缀
        for i, prefix_set in enumerate(prefix_sets):
            # 检查文件名是否以任一前缀开头
            for prefix in prefix_set:
                if filename.startswith(prefix):
                    dest_path = os.path.join(dest_dirs[i], filename)
                    shutil.copy2(filepath, dest_path)   # 复制并保留元数据
                    copied_counts[i] += 1
                    break   # 匹配到一个前缀即可，无需继续检查该文档的其他前缀

        # 每处理10个文件打印一次进度（可选）
        if idx % 10 == 0 or idx == total_files:
            print(f"进度：{idx}/{total_files}")

    # 输出各目标文件夹的复制结果
    for i, label in enumerate(dest_labels):
        print(f"{label}：共复制 {copied_counts[i]} 个文件到 {dest_dirs[i]}")

if __name__ == '__main__':
    # 加载三个前缀集合
    prefixes1 = load_prefixes(PREFIX_FILE1)
    prefixes2 = load_prefixes(PREFIX_FILE2)
    prefixes3 = load_prefixes(PREFIX_FILE3)

    print(f"文档1加载了 {len(prefixes1)} 个前缀")
    print(f"文档2加载了 {len(prefixes2)} 个前缀")
    print(f"文档3加载了 {len(prefixes3)} 个前缀")

    # 组织数据
    dest_dirs = [DEST_DIR1, DEST_DIR2, DEST_DIR3]
    prefix_sets = [prefixes1, prefixes2, prefixes3]
    dest_labels = ["文档1", "文档2", "文档3"]

    # 执行复制
    copy_matching_files(SOURCE_DIR, dest_dirs, prefix_sets, dest_labels)

    print("所有复制任务完成！")