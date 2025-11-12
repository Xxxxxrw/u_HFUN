# 大文件夹中包含多个小文件夹，每个小文件夹包含100张图片，从每个小文件夹中随机不重复的选取90张
import os
import random
import shutil

# 设置大文件夹路径和选取图片数量
big_folder_path = "/home/zjl0601/dataset/Images"
num_images = 50

# 获取大文件夹中所有小文件夹的路径
small_folder_paths = [os.path.join(big_folder_path, folder) for folder in os.listdir(big_folder_path)]

# 对于每个小文件夹执行操作
for small_folder_path in small_folder_paths:
    # 获取当前小文件夹中所有图片的文件名列表
    image_files = [f for f in os.listdir(small_folder_path) if f.endswith(".tif")]

    # 随机选择不重复的图片
    selected_images = random.sample(image_files, num_images)

    # 可选：将选取的图片从当前小文件夹移动到目标文件夹
    target_folder = "/home/zjl0601/dataset/UCMerge_test"
    os.makedirs(target_folder, exist_ok=True)
    for image_file in selected_images:
        src_path = os.path.join(small_folder_path, image_file)
        dst_path = os.path.join(target_folder, image_file)
        shutil.move(src_path, dst_path)

# 输出每个小文件夹选取的图片列表
# for i, small_folder_path in enumerate(small_folder_paths):
#     selected_images = os.listdir(os.path.join(target_folder, f"small_folder_{i + 1}"))
#     print(f"Selected Images in small_folder_{i + 1}:", selected_images)
