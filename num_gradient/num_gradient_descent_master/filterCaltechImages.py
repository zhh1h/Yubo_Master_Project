from PIL import Image
import numpy as np
import sys
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels
import  os
import torch
from collections import Counter
import cv2

#from t_ngd_cifar10 import net
#from t_ngd_cifar10 import preprocess_image
#from t_ngd_cifar10 import preprocess_with_transform_fn



# # 预处理图像函数
# def preprocess_image(h, w, x):
#     x = x.astype('uint8')
#     pixels = x.reshape((h, w, 3))
#     img = Image.fromarray(pixels, mode='RGB')
#     img_tensor = torch.Tensor(np.array(img)).permute(2, 0, 1) / 255.0
#     return img_tensor

# if not os.path.exists('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/filterCaltechImages'):
#     os.makedirs('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/filterCaltechImages')


# input_folder_path = '/home/yubo/yubo_tem_code/knockoffnets/data/256_ObjectCategories/030.canoe'  # 要处理的图片所在的文件夹路径
# output_folder_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/filterCaltechImages'

# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)



input_folder_path = '/home/yubo/yubo_tem_code/knockoffnets/data/256_ObjectCategories/113.hummingbird'  # 要处理的图片所在的文件夹路径
output_folder_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/filterCaltechImages'
expected_class_name = 'bird'  # 预期的分类名称

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def classify_and_resize_images(input_folder, output_folder, expected_class):
    image_info = []  # 用于存储每张图片的分类和处理后的对象

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # 支持多种图片格式
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)

            img_pil = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
            h, w, img_array = linearize_pixels(img_pil)
            predicted_class = test_classifier(h, w, img_array)

            # 将图片信息（包括分类、图片对象和文件名）添加到列表中
            image_info.append((predicted_class, img_pil, filename))

    # 统计最常见的分类
    classifications = [info[0] for info in image_info]
    class_counter = Counter(classifications)
    most_common_class, _ = class_counter.most_common(1)[0]

    saved_images_count = 0  # 保存的图片计数器
    saved_expected_class_images_count = 0  # 保存的预期分类图片计数器
    saved_most_common_class_images_count = 0  # 保存的最常见分类的图片计数器

    # 保存最常见分类和预期分类的图片
    for predicted_class, img_pil, filename in image_info:
        if predicted_class == most_common_class or predicted_class == expected_class:
            output_image_name = f"{filename.split('.')[0]}_{predicted_class}.png"
            output_image_path = os.path.join(output_folder, output_image_name)
            img_pil.save(output_image_path, format='PNG')
            saved_images_count += 1
            if predicted_class == most_common_class:
                saved_most_common_class_images_count += 1
            if predicted_class == expected_class:
                saved_expected_class_images_count += 1

    print(f"最常见的分类: {most_common_class}")
    print(f"预期的分类: {expected_class}")
    print(f"总共保存的图片数量: {saved_images_count}")
    print(f"保存的最常见类别的图片数量：{saved_most_common_class_images_count}")
    print(f"保存的预期分类的图片数量: {saved_expected_class_images_count}")

classify_and_resize_images(input_folder_path, output_folder_path, expected_class_name)
