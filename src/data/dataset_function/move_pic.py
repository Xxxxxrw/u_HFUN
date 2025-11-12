import os
import random
import shutil

# 指定原始文件夹和目标文件夹路径
source_folder = '/home/zjl0601/dataset/UCMerge_train'
target_folder = '/home/zjl0601/dataset/UCMerge_valid'

# 获取原始文件夹中所有图片的文件名列表
image_files = [f for f in os.listdir(source_folder) if f.endswith('.tif') or f.endswith('.png')]

# 随机选取210张不重复的图片
selected_images = random.sample(image_files, 210)

# 移动选取的图片到目标文件夹
for image_name in selected_images:
    source_path = os.path.join(source_folder, image_name)
    target_path = os.path.join(target_folder, image_name)
    shutil.move(source_path, target_path)
