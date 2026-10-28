import os
import glob
import shutil

# ===== 请修改此处为您的输出总目录 =====
ROOT_DIR = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/train/0_normal"  # 例如：/home/hyw/dataset/chendawei/train-classification/ICPR-2014/


# =====================================

def delete_bg_samples(root_dir):
    pattern = os.path.join(root_dir, "**", "*_neg_bg_*.png")  # 匹配所有子目录下的纯背景图
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("未找到任何纯背景样本（*_neg_bg_*.png），无需删除。")
        return

    print(f"找到 {len(files)} 个纯背景样本文件：")
    for f in files[:10]:  # 只显示前10个，避免刷屏
        print(f"  {f}")
    if len(files) > 10:
        print(f"  ... 还有 {len(files) - 10} 个文件")

    # 安全确认
    confirm = input(f"\n确定要永久删除这 {len(files)} 个文件吗？(输入 yes 继续): ")
    if confirm.lower() != "yes":
        print("操作已取消。")
        return

    deleted = 0
    for file_path in files:
        try:
            os.remove(file_path)
            deleted += 1
        except Exception as e:
            print(f"删除失败: {file_path}，错误: {e}")

    print(f"成功删除 {deleted} 个纯背景样本文件。")


if __name__ == "__main__":
    delete_bg_samples(ROOT_DIR)