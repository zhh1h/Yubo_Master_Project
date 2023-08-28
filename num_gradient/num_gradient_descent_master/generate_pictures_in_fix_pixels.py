from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from t_ngd_cifar10 import test_classifier, linearize_pixels
from t_ngd_cifar10 import create_f

# 创建保存图片的文件夹，如果不存在的话
if not os.path.exists('fix_pixels_images'):
    os.makedirs('fix_pixels_images')

# 准备用于展示图片的 matplotlib 子图
fig, axes = plt.subplots(1, 10, figsize=(20, 2))

# 生成10张32*32的单色图片
for i, pixel_value in enumerate(range(137, 127, -1)):
    # 创建一个32*32*3的NumPy数组
    img_array = np.full((32, 32, 3), pixel_value, dtype=np.uint8)

    # 使用PIL从NumPy数组创建一个图片对象
    img = Image.fromarray(img_array, 'RGB')

    # 保存图片
    img_path = f'fix_pixels_images/fix_pixels{pixel_value}.jpg'
    img.save(img_path)
    h, w, img_array = linearize_pixels(Image.fromarray(np.uint8(img)))
    identified_class = test_classifier(h, w, img_array,return_class_index = True,return_confidence = True)

    # 展示图片
    axes[i].imshow(img)
    axes[i].set_title(f'Pixel: {pixel_value}')
    axes[i].axis('off')

plt.show()
print("图片生成和保存完成！")

