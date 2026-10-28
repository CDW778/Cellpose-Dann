import os
import glob
import shutil
from pathlib import Path

# ==================== 用户配置 ====================
ERROR_DIR = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/cellpose_errors/errors-img"          # 存放 *_errors.png 的文件夹
ORIGINAL_IMAGE_DIR = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/original-img"  # 原始 .tiff 所在文件夹
OUTPUT_DIR = "/home/hyw/dataset/chendawei/train-classification/ICPR-2014/cellpose_errors/errors-original-img"    # 提取出的原图保存位置
LOG_FILE = os.path.join(OUTPUT_DIR, "extract_log.txt")

# ==================================================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 扫描所有 error 图片
    error_pattern = os.path.join(ERROR_DIR, "*_errors.png")
    error_files = glob.glob(error_pattern)
    if not error_files:
        print(f"在 {ERROR_DIR} 中未找到任何 *_errors.png 文件。")
        return

    print(f"发现 {len(error_files)} 个漏检标记文件。")

    # 打开日志
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        log.write("=== 提取漏检原图日志 ===\n")
        success_count = 0
        fail_count = 0

        for err_path in error_files:
            # 获取基础名，例如 A03_00Ca
            basename = os.path.basename(err_path)          # "A03_00Ca_errors.png"
            base_core = basename.replace("_errors.png", "") # "A03_00Ca"

            # 在原始图像目录中查找 .tiff 或 .tif
            tiff_candidates = [
                os.path.join(ORIGINAL_IMAGE_DIR, f"{base_core}.tiff"),
                os.path.join(ORIGINAL_IMAGE_DIR, f"{base_core}.tif"),
                os.path.join(ORIGINAL_IMAGE_DIR, f"{base_core}.TIF"),
                os.path.join(ORIGINAL_IMAGE_DIR, f"{base_core}.TIFF"),
            ]
            found_path = None
            for candidate in tiff_candidates:
                if os.path.exists(candidate):
                    found_path = candidate
                    break

            if found_path:
                # 复制到输出目录
                dest_path = os.path.join(OUTPUT_DIR, os.path.basename(found_path))
                shutil.copy2(found_path, dest_path)  # copy2 保留元数据
                log.write(f"✅ 成功: {base_core}  -> {dest_path}\n")
                success_count += 1
                print(f"已复制: {base_core}")
            else:
                log.write(f"❌ 未找到: {base_core} (在 {ORIGINAL_IMAGE_DIR} 中无对应 .tiff/.tif)\n")
                fail_count += 1
                print(f"警告: 未找到 {base_core} 的原图")

        # 汇总
        log.write(f"\n总计: 成功 {success_count} 个, 失败 {fail_count} 个\n")
        print(f"\n处理完成。成功复制 {success_count} 个原图，{fail_count} 个未找到。")
        print(f"日志保存在: {LOG_FILE}")

if __name__ == "__main__":
    main()