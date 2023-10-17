# import matplotlib.pyplot as plt
# from PIL import Image
# import os
#
# def check_image_size(directory, width=32, height=32):
#     mismatched_images = []
#
#     # 遍历指定目录的所有文件
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 filepath = os.path.join(root, file)
#                 with Image.open(filepath) as img:
#                     if img.size != (width, height) or img.mode != 'RGB':
#                         mismatched_images.append((filepath, img.size))
#
#     return mismatched_images
#
# def resize_to_target(image_path, output_path, target_width=32, target_height=32):
#     with Image.open(image_path) as img:
#         # Ensure image is RGB
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         # Resize the image
#         img_resized = img.resize((target_width, target_height), Image.ANTIALIAS)
#         img_resized.save(output_path)
#
# directory_path = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/std_deviation"  # 替换为您的文件夹路径
# mismatched_info = check_image_size(directory_path)
#
# print(f"总共有 {len(mismatched_info)} 张图片的尺寸不是 32x32 或 不是 RGB 模式.")
#
# for img_path, size in mismatched_info:
#     print(f"图片 {img_path} 的尺寸为 {size[0]}x{size[1]}")
#
# # Resize the mismatched images
# for img_path, _ in mismatched_info:
#     resize_to_target(img_path, img_path)  # Overwriting the original images
#
# print("不符合尺寸的图片已被调整为 32x32x3.")
#
#
# image_path_0 = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/std_deviation/065_0006.jpg"
#
# # 打开并显示图像
# # 使用Pillow库打开图像
# img = Image.open(image_path_0)
#
# # 使用matplotlib显示图像
# plt.imshow(img)
# plt.axis('off')  # 不显示坐标轴
# plt.show()

import os
import cv2
import random


def random_resize_images(folder_a, folder_b, m):
    # 获取文件夹a中的所有jpg图片
    image_files = [f for f in os.listdir(folder_a) if f.endswith('.jpg')]

    # 从列表中随机选择m个图片
    selected_images = random.sample(image_files, m)

    # 确保输出文件夹存在
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)

    # 对每个选中的图片执行操作
    for idx, image_file in enumerate(selected_images, 1):
        # 读取图片
        img_path = os.path.join(folder_a, image_file)
        img = cv2.imread(img_path)

        # 使用双线性插值缩放图片到32x32像素
        resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)

        # 保存图片到文件夹b，文件名为文件夹a的名称 + 序号，格式为png
        output_filename = os.path.basename(folder_a) + '_' + str(idx) + '.png'
        output_path = os.path.join(folder_b, output_filename)
        cv2.imwrite(output_path, resized_img)


# 使用方法
folder_a = '/home/yubo/yubo_tem_code/knockoffnets/data/256_ObjectCategories/252.car-side-101'
folder_b = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/std_deviation'
m = 52 # 你想要的图片数量
random_resize_images(folder_a, folder_b, m)
