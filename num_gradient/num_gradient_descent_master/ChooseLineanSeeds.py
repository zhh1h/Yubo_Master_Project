import os
import shutil
import numpy as np
from PIL import Image
import json
import sys
from collections import defaultdict
import logging

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

def find_top_images(directory, output_directory, score_threshold=0.7):
    top_images = defaultdict(lambda: [])

    # 遍历文件夹中的所有图片
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                h, w, img_array = linearize_pixels(img)
                cls, score = test_classifier(h, w, img_array.flatten(), return_confidence=True)

                # 只保留得分超过指定阈值的图片
                if score > score_threshold:
                    top_images[cls].append((score, filename))

    # 移动得分超过阈值的图片到指定文件夹
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for cls, images in top_images.items():
        for score, filename in images:
            src_path = os.path.join(directory, filename)
            # 新文件名，添加分类种类信息
            new_filename = f'{os.path.splitext(filename)[0]}_class_{cls}{os.path.splitext(filename)[1]}'
            dst_path = os.path.join(output_directory, new_filename)
            shutil.move(src_path, dst_path)  # 使用move方法移动文件

    print("图片已成功分类和移动.")

# 调用函数
find_top_images('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/std_deviation', '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/random_pattern_linear_interpolation_top3')
