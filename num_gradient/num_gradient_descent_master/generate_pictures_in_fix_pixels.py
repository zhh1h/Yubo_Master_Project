from PIL import Image
import numpy as np
import os

# 创建保存图片的文件夹，如果不存在的话
if not os.path.exists('fix_pixels_images'):
    os.makedirs('fix_pixels_images')

# 生成10张32*32的单色图片
for pixel_value in range(127, 117, -1):
    # 创建一个32*32*3的NumPy数组
    img_array = np.full((32, 32, 3), pixel_value, dtype=np.uint8)

    # 使用PIL从NumPy数组创建一个图片对象
    img = Image.fromarray(img_array, 'RGB')

    # 保存图片
    img.save(f'fix_pixels_images/fix_pixels{pixel_value}.png')

print("图片生成和保存完成！")
