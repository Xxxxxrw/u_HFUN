# 将大文件夹中所有小文件夹中包含的jpg图片转换成png
import os
import shutil

def change_file_extension(folder_path):
    # 遍历NWPU-RESISC45文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 判断文件是否为JPG格式
            if file.endswith(".jpg"):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                # 构建新的文件路径，将JPG扩展名替换为PNG
                new_file_path = file_path[:-4] + ".png"
                # 重命名文件
                shutil.move(file_path, new_file_path)

# 指定NWPU-RESISC45文件夹路径
folder_path = "/home/zjl0601/test"

# 调用函数进行文件后缀更改
change_file_extension(folder_path)
