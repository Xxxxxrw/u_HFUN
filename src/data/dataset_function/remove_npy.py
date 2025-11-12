import os

folder_path = '/home/zjl0601/dataset/UCMerge_valid/HR'  # 文件夹路径

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)

# 遍历文件列表
for file_name in file_list:
    # 检查文件后缀是否为 '.npy'
    if file_name.endswith('.npy'):
        # 构造文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 删除文件
        os.remove(file_path)
