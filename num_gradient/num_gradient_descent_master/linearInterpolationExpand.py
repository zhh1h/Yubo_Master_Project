import os
import numpy as np
from PIL import Image
import json
import sys
import logging

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

# 假设你的网络和类别列表在这里初始化

def linear_interpolation(img1, img2, alpha):
    return img1 * (1 - alpha) + img2 * alpha

def parse_image_info(filename):
    # 假设文件名格式为 "n02085936_6348_dog.png"
    parts = filename.split('_')
    img_id = parts[1]
    img_class = parts[2].split('.')[0]
    return img_id, img_class

def generate_interpolated_images(h, w, start_img, start_img_filename, end_img, end_img_filename, steps, images_save_path):
    start_img_id, start_class = parse_image_info(start_img_filename)
    end_img_id, end_class = parse_image_info(end_img_filename)

    step_size = 1 / steps
    alpha_values = np.linspace(0, 1, steps + 1)  # Ensure include 0 and 1

    results = []
    for alpha in alpha_values:
        interpolated_img_array = linear_interpolation(np.asarray(start_img), np.asarray(end_img), alpha)
        interpolated_img = Image.fromarray(np.clip(interpolated_img_array, 0, 255).astype(np.uint8), 'RGB')
        interpolated_class = test_classifier(*linearize_pixels(interpolated_img), return_confidence=False)

        interpolated_image_filename = f"{start_img_id}_{end_img_id}_{start_class}_{end_class}_{alpha:.2f}_{interpolated_class}.png"
        interpolated_image_path = os.path.join(images_save_path, interpolated_image_filename)
        interpolated_img.save(interpolated_image_path)

        results.append((interpolated_class, interpolated_image_path))

    return results

# Example usage
start_img_filename = 'n02085936_6348_dog.png'
end_img_filename = 'Image_6_horse.png'
start_img = Image.open(os.path.join('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/chooseLineanSeeds', start_img_filename))
end_img = Image.open(os.path.join('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/chooseLineanSeeds', end_img_filename))

images_save_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/linearinterpolationExpand'
steps = 50

# Generate interpolated images and get their classification results
interpolated_results = generate_interpolated_images(32, 32, start_img, start_img_filename, end_img, end_img_filename, steps, images_save_path)