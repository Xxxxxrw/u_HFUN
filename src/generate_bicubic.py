#
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# 1. 用户配置区域 - 您只需要修改这里
# ==============================================================================

# 您的 Benchmark 数据集根目录
# 例如: '/home/xingrenwang2024/EDSR-PyTorch/src/srdataset/benchmark'
BENCHMARK_ROOT = Path('/home/xingrenwang2024/EDSR-PyTorch/src/srdataset/benchmark')

# 您想要的上采样倍数 (例如 2, 3, 4)
SCALE_FACTOR = 4

# 结果输出的根目录 (脚本会自动创建)
OUTPUT_ROOT = Path('./bicubic_results')


# ==============================================================================
# 脚本主逻辑 - 一般无需修改
# ==============================================================================

def bicubic_upsample():
    """
    对SISR基准数据集的LR图像进行Bicubic上采样。
    """
    # 定义五个标准的基准数据集名称
    dataset_names = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

    print(f"开始进行 Bicubic 上采样，倍数: X{SCALE_FACTOR}")
    print(f"输入数据根目录: {BENCHMARK_ROOT}")
    print(f"输出结果根目录: {OUTPUT_ROOT}")
    print("-" * 50)

    # 遍历所有指定的数据集
    for dataset in dataset_names:
        # 1. 构建输入路径，根据您提供的标准格式
        # 例如: .../benchmark/Set5/LR_bicubic/X2
        lr_dir = BENCHMARK_ROOT / dataset / 'LR_bicubic' / f'X{SCALE_FACTOR}'

        # 2. 检查输入目录是否存在
        if not lr_dir.is_dir():
            print(f"警告: 目录 '{lr_dir}' 不存在，跳过数据集 '{dataset}'。")
            continue

        # 3. 构建输出路径
        # 例如: ./bicubic_results/Set5/X2
        output_dir = OUTPUT_ROOT / dataset / f'X{SCALE_FACTOR}'

        # 4. 创建输出目录 (如果不存在)
        # exist_ok=True 表示如果目录已存在，则不会报错
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n正在处理数据集: {dataset} (X{SCALE_FACTOR})")
        print(f"从: {lr_dir}")
        print(f"到: {output_dir}")

        # 5. 查找所有支持的图像文件
        # 使用 glob 匹配多种常见图像格式
        image_files = sorted(list(lr_dir.glob('*.png')) + list(lr_dir.glob('*.jpg')) + list(lr_dir.glob('*.bmp')))

        if not image_files:
            print(f"警告: 在 '{lr_dir}' 中未找到任何图像文件。")
            continue

        # 6. 遍历并处理每张图像
        for img_path in tqdm(image_files, desc=f"Upsampling {dataset}"):
            try:
                # 打开低分辨率图像
                with Image.open(img_path) as lr_img:
                    # 获取原始尺寸
                    width, height = lr_img.size
                    # 计算目标尺寸
                    new_width = width * SCALE_FACTOR
                    new_height = height * SCALE_FACTOR

                    # 使用 Bicubic 插值进行上采样
                    # Image.Resampling.BICUBIC 是 Pillow 9.0+ 的推荐用法
                    # 旧版本使用 Image.BICUBIC
                    hr_img = lr_img.resize((new_width, new_height), Image.Resampling.BICUBIC)

                    # 构建输出文件的完整路径
                    output_path = output_dir / img_path.name

                    # 保存上采样后的图像
                    hr_img.save(output_path)

            except Exception as e:
                print(f"\n处理文件 {img_path.name} 时出错: {e}")

    print("\n" + "=" * 50)
    print("所有数据集处理完毕！")
    print(f"结果已保存至: {OUTPUT_ROOT}")


if __name__ == '__main__':
    bicubic_upsample()