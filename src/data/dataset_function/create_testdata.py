import os
import shutil

# 定义大文件夹和目标文件夹的路径
source_folder = '/home/zjl0601/dataset/UC_10'  # 大文件夹路径
target_folder = '/home/zjl0601/dataset/benchmark/uc10'  # 目标文件夹路径

# 遍历大文件夹下的所有子文件夹
for root, dirs, files in os.walk(source_folder):
    # 遍历当前子文件夹中的所有文件
    for file in files:
        # 构建源文件的路径和目标文件的路径
        source_path = os.path.join(root, file)
        target_path = os.path.join(target_folder, file)

        # 复制文件到目标文件夹中
        shutil.copy(source_path, target_path)