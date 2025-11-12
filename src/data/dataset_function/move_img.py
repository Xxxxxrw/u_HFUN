import os
import shutil


def extract_images_from_folders(folder_path, target_folder):
    """
    从大文件夹中的所有小文件夹中提取图片，并复制到指定的目标文件夹中。

    参数：
    - folder_path: 大文件夹的路径，包含多个小文件夹。
    - target_folder: 目标文件夹的路径，用于存储提取的图片。
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)

                if file.endswith(('.jpg', '.jpeg', '.png','.tif')):
                    target_file_path = os.path.join(target_folder, file)
                    shutil.copy(file_path, target_file_path)

folder_path = '/home/zjl0601/dataset/AID'
target_folder = '/home/zjl0601/dataset/demo'

extract_images_from_folders(folder_path, target_folder)
