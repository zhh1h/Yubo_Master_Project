from PIL import Image
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import os
import torch

# 添加路径
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

# 保存噪声图像的文件夹
SAVE_PATH = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/std_0.2'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 定义转换函数
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

from torchvision.transforms import ToPILImage

def generate_image_with_noise_and_classify(h, w, img_array, img_name, std_deviation, original_class):
    original_image = img_array.reshape((h, w, 3)).astype('uint8')

    # 生成高斯噪声并添加到原图
    noise = np.random.normal(0, std_deviation, original_image.shape)
    new_image = original_image + noise
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    # 对新图像进行分类并获取置信分数
    predicted_class, scores = test_classifier(h, w, new_image, return_all_scores=True)
    print(f"New image class: {predicted_class}, Scores: {scores}")

    # 保存图像
    img_path = os.path.join(SAVE_PATH, f"{img_name}_{predicted_class}_{std_deviation}.png")
    new_image_pil = Image.fromarray(new_image, 'RGB')
    new_image_pil = transform_fn(new_image_pil)  # 使用 transform_fn 进行预处理
    to_pil_image = ToPILImage()
    new_image_pil = to_pil_image(new_image_pil)  # 将张量转换为 PIL 图像
    new_image_pil.save(img_path)  # 保存预处理后的图片

    return predicted_class, scores

# 图像文件夹的路径
folder_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/_0_images'

for img_file in os.listdir(folder_path):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, img_file)
        your_original_image = Image.open(img_path).convert("RGB")
        h, w, img_array = linearize_pixels(your_original_image)

        # 获取原图像的分类
        original_class, original_scores = test_classifier(h, w, img_array, return_all_scores=True)
        print(f"Original image class: {original_class}, Scores: {original_scores}")

        # 初始化标准差
        std_deviation = 0

        # 循环以找出导致分类改变的最小标准差
        while True:
            print(f"Testing standard deviation: {std_deviation}")
            new_class, new_scores = generate_image_with_noise_and_classify(h, w, img_array, os.path.splitext(img_file)[0], std_deviation, original_class)

            # 如果类别改变，处理下一张图片
            if new_class != original_class:
                print(f"When the standard deviation is {std_deviation}, {img_file}'s class changes to {new_class}. Ending {img_file}'s operation.")
                break  # 结束循环

            std_deviation += 0.2
