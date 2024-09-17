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

def find_top_images(directory, output_directory):
    top_images = defaultdict(lambda: [])

    # 遍历文件夹中的所有图片
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                h, w, img_array = linearize_pixels(img)
                cls, score = test_classifier(h, w, img_array.flatten(), return_confidence=True)

                # 保存每个类别得分最高的三张图片
                if len(top_images[cls]) < 3:
                    top_images[cls].append((score, filename))
                    top_images[cls].sort(reverse=True, key=lambda x: x[0])  # 保证最高分在前
                else:
                    if score > top_images[cls][-1][0]:  # 如果当前得分超过第三高的得分
                        top_images[cls][-1] = (score, filename)
                        top_images[cls].sort(reverse=True, key=lambda x: x[0])  # 重新排序

    # 保存得分最高的图片到指定文件夹
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for cls, images in top_images.items():
        for _, filename in images:
            src_path = os.path.join(directory, filename)
            dst_path = os.path.join(output_directory, filename)
            shutil.copy(src_path, dst_path)  # 使用copy方法复制文件

    print("图片已成功分类和复制.")

# 调用函数
find_top_images('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/random_pattern_linear_interpolation_0.7', '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/filter_random_pattern_linear_interpolation_top3')