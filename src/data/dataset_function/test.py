import os
import glob

folder_path = '/home/zjl0601/test'  # 替换为目标文件夹的路径
def rename_images(folder_path):
    # 获取文件夹中所有图片文件的路径
    image_files = glob.glob(os.path.join(folder_path, '*.png'))  # 可根据需要修改文件扩展名
    for i, file_path in enumerate(image_files):
        # 构造新的文件名
        print(i)
        new_name = f"{i+1}.png"  # 可根据需要修改文件扩展名

        # 重命名文件
        new_path = os.path.join(folder_path, new_name)
        os.rename(file_path, new_path)

# 调用示例


rename_images(folder_path)
