import os
from PIL import Image

# 指定文件夹路径
folder_path = '/home/tyh123456/dataset/DIV2K_test/DIV2K_train_HR'

# 遍历文件夹中的所有文件
index = 1
for filename in os.listdir(folder_path):
    # 判断文件是否为图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建图片的完整路径
        image_path = os.path.join(folder_path, filename)

        # 打开图片
        image = Image.open(image_path)

        # 旋转图片
        rotated_image_90 = image.rotate(90)
        rotated_image_180 = image.rotate(180)
        rotated_image_270 = image.rotate(270)

        # 保存旋转后的图片
        rotated_image_90.save(os.path.join(folder_path, 'rotated_90_' + filename))
        rotated_image_180.save(os.path.join(folder_path, 'rotated_180_' + filename))
        rotated_image_270.save(os.path.join(folder_path, 'rotated_270_' + filename))

        # 关闭图片
        image.close()
        print(f'done {index}')
        index += 1
