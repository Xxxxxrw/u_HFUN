# 在测试集中创建两个文件夹,HR和LR_bicubic
# import os
# import shutil
#
# # 大文件夹路径
# big_folder = "/home/zjl0601/dataset/benchmark/UC_10"
#
# # 获取大文件夹下的所有小文件夹
# sub_folders = [f for f in os.listdir(big_folder) if os.path.isdir(os.path.join(big_folder, f))]
#
# # 创建名为"shi"和"hello"的新文件夹
# for sub_folder in sub_folders:
#     sub_folder_path = os.path.join(big_folder, sub_folder)
#     new_folder1 = os.path.join(sub_folder_path, "HR")
#     new_folder2 = os.path.join(sub_folder_path, "LR_bicubic")
#     os.makedirs(new_folder1, exist_ok=True)
#     os.makedirs(new_folder2, exist_ok=True)

# 大文件夹下包含很多小文件夹，在所有的小文件夹中，在小文件夹中LR_bicubie的文件夹中，创建三个文件夹，名字分别为X2、X3和X4
import os

# 大文件夹路径
big_folder = "/home/zjl0601/dataset/benchmark/UC_10"

# 获取大文件夹下的所有小文件夹
sub_folders = [f for f in os.listdir(big_folder) if os.path.isdir(os.path.join(big_folder, f))]

# 遍历每个小文件夹
for sub_folder in sub_folders:
    sub_folder_path = os.path.join(big_folder, sub_folder)

    # 在小文件夹的"HR"文件夹中创建子文件夹
    hr_folder = os.path.join(sub_folder_path, "LR_bicubic")
    for scale in ["X2", "X3", "X4"]:
        scale_folder = os.path.join(hr_folder, scale)
        os.makedirs(scale_folder, exist_ok=True)

