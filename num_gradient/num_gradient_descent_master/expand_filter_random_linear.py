import os
import numpy as np
from PIL import Image
import json
import sys
from collections import defaultdict
import logging

sys.path.append(
    "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels
import torch
print(torch.version.cuda)

# 设置日志文件
logging.basicConfig(filename='/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/decision_linear_random_30_100_log.txt', level=logging.INFO)


def interpolate_images_in_directory(directory, output_directory, all_interpolations_dir, step=100):
    images_by_class = defaultdict(lambda: [])

    # 读取文件夹中的所有图片并按类别分类
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                img = img.convert('RGB')  # 确保图像是 RGB 格式
                h, w, img_array = linearize_pixels(img)
                print(f'Processing {filename}: shape={(h, w, 3)}')  # 调试输出
                cls, score = test_classifier(h, w, img_array, return_confidence=True)
                images_by_class[cls].append((filename, h, w, img_array))

    # 创建保存生成的全部插值图片的目录
    if not os.path.exists(all_interpolations_dir):
        os.makedirs(all_interpolations_dir)

    # 进行线性插值并保存决策边缘的图像
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for cls_a, images_a in images_by_class.items():
        for cls_b, images_b in images_by_class.items():
            if cls_a != cls_b:
                for filename_a, h_a, w_a, img_array_a in images_a:
                    for filename_b, h_b, w_b, img_array_b in images_b:
                        interpolate_and_save(h_a, w_a, img_array_a, h_b, w_b, img_array_b, cls_a, cls_b, filename_a,
                                             filename_b, output_directory, all_interpolations_dir, step)

    print("图片插值处理完成并保存.")


def interpolate_and_save(h_a, w_a, img_array_a, h_b, w_b, img_array_b, cls_a, cls_b, filename_a, filename_b,
                         output_directory, all_interpolations_dir, step=100):
    assert (h_a, w_a) == (h_b, w_b), "Images must have the same shape for interpolation"

    c = 3  # 确保图像是RGB格式的三通道

    img_array_a = img_array_a.reshape((h_a, w_a, c))
    img_array_b = img_array_b.reshape((h_b, w_b, c))

    prev_cls = cls_a  # 假设开始时的分类是 cls_a

    for alpha in np.linspace(0, 1, step):
        img_array_c = (1 - alpha) * img_array_a + alpha * img_array_b
        img_array_c = img_array_c.astype(np.uint8)
        img_c = Image.fromarray(img_array_c)

        # 保存所有的插值图像
        base_filename_a = os.path.splitext(filename_a)[0]
        base_filename_b = os.path.splitext(filename_b)[0]
        interpolation_filename = f'interpolated_{base_filename_a}_and_{base_filename_b}_alpha_{alpha:.2f}.png'
        interpolation_path = os.path.join(all_interpolations_dir, interpolation_filename)
        img_c.save(interpolation_path)

        # 判断插值图像的分类
        cls_c, score = test_classifier(h_a, w_a, img_array_c.flatten(), return_confidence=True)

        # 检查分类是否发生变化
        if cls_c != prev_cls:
            # 保存处于决策边缘的图像
            decision_edge_filename = f'interpolated_{base_filename_a}_and_{base_filename_b}_alpha_{alpha:.2f}_class_{cls_c}.png'
            decision_edge_path = os.path.join(output_directory, decision_edge_filename)
            img_c.save(decision_edge_path)
            print(f'保存插值图像: {decision_edge_path}')

            # 记录决策边缘的图像ID
            logging.info(
                f'Decision edge image saved: {decision_edge_filename}, Alpha: {alpha}, Classes: {cls_a} -> {cls_c}')

        prev_cls = cls_c


# 调用函数
interpolate_images_in_directory(
    '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/filter_random_pattern_linear_interpolation_top3',
    '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/interpolated_linear_random_30_100',
    '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/all_interpolations_30_100'
)

# 调用函数

