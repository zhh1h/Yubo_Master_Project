from PIL import Image
import numpy as np
import sys
import re
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
#from cifar10vgg19testClassifier import test_classifier, linearize_pixels
import  os
import torch
from collections import Counter
import cv2
from collections import defaultdict
from shutil import move


def remove_excess_images_with_regex(directory, keywords, max_images=256):
    # 编译正则表达式模式，匹配以下划线开头，英文句号结尾的关键词
    patterns = {keyword: re.compile(r'_' + re.escape(keyword) + r'\.') for keyword in keywords}

    # 存储匹配到的文件
    matched_files = defaultdict(list)

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            for keyword, pattern in patterns.items():
                if pattern.search(filename):
                    matched_files[keyword].append(filename)
                    break  # 匹配到一个关键词后就不再继续匹配其他关键词

    # 删除每个关键词超过max_images数量的图片
    for keyword, filenames in matched_files.items():
        if len(filenames) > max_images:
            # 超过指定数量的图片进行删除
            filenames_to_remove = sorted(filenames)[max_images:]
            for filename in filenames_to_remove:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Removed {file_path}")

# 定义关键词列表
keywords = ["ship"]

# 调用函数
remove_excess_images_with_regex('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/afterFilterCaltechImages', keywords)