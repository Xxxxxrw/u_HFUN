import os
from PIL import Image

def downsample_images(folder_path, scale_factor=2):
    # 获取文件夹中所有图片文件的路径
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in image_files:
        # 构造图片的完整路径
        image_path = os.path.join(folder_path, image_file)

        # 打开图片
        image = Image.open(image_path)

        # 计算新的宽度和高度
        width, height = image.size
        new_width = width // scale_factor
        new_height = height // scale_factor

        # 进行双三次下采样
        downscaled_image = image.resize((new_width, new_height), resample=Image.BICUBIC)

        # 保存下采样后的图片
        downscaled_image.save(image_path)

        # 关闭图片
        image.close()
        downscaled_image.close()

# 调用示例
folder_path = '/home/tyh123456/dataset/DF2K/DF2K_train_LR_bicubic/X2'  # 替换为目标文件夹的路径
downsample_images(folder_path, scale_factor=2)
