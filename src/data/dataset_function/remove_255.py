import os
from PIL import Image

# 文件夹路径
folder_path = "/home/zjl0601/dataset/UCMerge_test/LR_x3"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 仅处理图像文件
    if filename.endswith((".tif", ".jpeg", ".png")):
        # 图片完整路径
        image_path = os.path.join(folder_path, filename)

        # 打开图像
        image = Image.open(image_path)

        # 检查图像尺寸
        if image.size != (85, 85):
            # 删除不是256x256的图像文件
            os.remove(image_path)
            print(f"Removed {filename}")
