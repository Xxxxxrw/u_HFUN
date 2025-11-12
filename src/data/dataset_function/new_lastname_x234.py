import os

def add_suffix_to_png_files(folder_path):
    # 获取文件夹中所有 PNG 图片文件的路径
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for image_file in image_files:
        # 构造图片的完整路径
        image_path = os.path.join(folder_path, image_file)
        # 构造新的文件名，添加 "x2" 后缀
        new_image_file = image_file.replace('.png', 'x3.png')
        # 构造新的图片的完整路径
        new_image_path = os.path.join(folder_path, new_image_file)
        # 重命名图片文件
        os.rename(image_path, new_image_path)
# 调用示例
folder_path = '/home/tyh123456/dataset/DF2K/Flickr2K_LR_bicubic（复件）/X3'  # 替换为目标文件夹的路径
add_suffix_to_png_files(folder_path)
