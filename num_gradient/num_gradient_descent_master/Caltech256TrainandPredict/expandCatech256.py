from PIL import Image
import numpy as np
import sys
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")
from t_ngd_cifar10 import test_classifier, linearize_pixels
import os
import torch

# 保存噪声图像的文件夹
SAVE_PATH = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/caltech_std_deviation'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def generate_image_with_noise_and_classify(h, w, img_array, img_name, std_deviation):
    original_image = img_array.reshape((h, w, 3)).astype('uint8')

    # 生成高斯噪声并添加到原图
    noise = np.random.normal(0, std_deviation, original_image.shape)
    new_image = original_image + noise
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    # 对新图像进行分类
    h, w, img_array = linearize_pixels(new_image)
    predicted_class = test_classifier(h, w, img_array)
    print(f"New image class: {predicted_class}")

    # 如果类别与原始类别相同，则保存图像
    if predicted_class == original_class:
        img_path = os.path.join(SAVE_PATH, f"{img_name}_{predicted_class}_{std_deviation}.png")
        Image.fromarray(new_image, 'RGB').save(img_path)
    return predicted_class

# 图像文件夹的路径
folder_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/030'

for img_file in os.listdir(folder_path):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, img_file)
        your_original_image = Image.open(img_path).convert("RGB")
        h, w, img_array = linearize_pixels(your_original_image)

        # 获取原图像的分类
        original_class = test_classifier(h, w, img_array)
        print(f"Original image class: {original_class}")

        # 初始化标准差
        std_deviation = 0

        # 循环以找出导致分类改变的最小标准差
        while True:
            print(f"Testing standard deviation: {std_deviation}")
            new_class = generate_image_with_noise_and_classify(h, w, img_array, os.path.splitext(img_file)[0], std_deviation)

            if new_class != original_class:
                print(f"When the standard deviation is {std_deviation}, {img_file}'s class changes to {new_class}. Ending {img_file}'s operation.")
                break  # 结束循环

            std_deviation += 1


