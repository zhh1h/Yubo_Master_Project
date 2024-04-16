import os
import numpy as np
from PIL import Image
import json
import sys
from collections import defaultdict
import logging

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

def find_top_images(directory, output_directory):
    top_images = {}
    scores = defaultdict(float)

    # 遍历文件夹中的所有图片
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                h, w, img_array = linearize_pixels(img)
                cls, score = test_classifier(h, w, img_array.flatten(), return_confidence=True)


                # 检查当前类别的最高分数图片
                if score > scores[cls]:
                    scores[cls] = score
                    top_images[cls] = filename

    # 保存得分最高的图片到指定文件夹
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for cls, filename in top_images.items():
        src_path = os.path.join(directory, filename)
        dst_path = os.path.join(output_directory, f"{filename}")
        os.rename(src_path, dst_path)

    print("图片已成功分类和移动.")

# 调用函数
find_top_images('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/afterFilterCaltechImages', '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/chooseLineanSeeds')