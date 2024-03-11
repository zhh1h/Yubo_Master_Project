import os
import numpy as np
from PIL import Image
import json
import sys
import logging

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

# Configure logging
logging.basicConfig(filename='/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/epsilon20_0.9.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Program started.")

def check_path_and_permissions(path):
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        try:
            os.makedirs(path)
            print(f"Path created: {path}")
        except Exception as e:
            print(f"Failed to create path: {path}. Error: {e}")
    else:
        print(f"Path exists: {path}")
        try:
            test_file_path = os.path.join(path, "test_write.txt")
            with open(test_file_path, "w") as test_file:
                test_file.write("Test write permissions.")
            os.remove(test_file_path)
            print(f"Write permissions OK: {path}")
        except Exception as e:
            print(f"Write permissions denied: {path}. Error: {e}")

def generate_images_and_weights_with_normal_distribution(h, w, original_img_array, epsilon_range, original_class, images_save_path, filename, expansion_counts):
    different_class_count = 0
    for epsilon in epsilon_range:
        covariance_matrix = np.diag([epsilon] * 3)
        noise = np.random.multivariate_normal([0, 0, 0], covariance_matrix, (h, w))
        new_image_array = original_img_array.reshape((h, w, 3)) + noise
        new_image_array = np.clip(new_image_array, 0, 255).astype(np.uint8)

        new_image = Image.fromarray(new_image_array, 'RGB')
        new_image_path = os.path.join(images_save_path, f"{os.path.splitext(filename)[0]}_epsilon_{epsilon}.png")
        new_image.save(new_image_path)

        expansion_counts[filename] = expansion_counts.get(filename, 0) + 1
        predicted_class = test_classifier(h, w, new_image_array.flatten())
        if predicted_class != original_class:
            different_class_count += 1

    ratio = different_class_count / len(epsilon_range) if different_class_count > 0 else 0
    return (True, ratio) if different_class_count > 0 else (False, 0)

def load_completed_images(progress_path):
    try:
        with open(progress_path, "r") as file:
            completed_images = set(file.read().splitlines())
    except FileNotFoundError:
        completed_images = set()
    return completed_images

def update_completed_images(progress_path, filename, completed_images):
    completed_images.add(filename)
    with open(progress_path, "w") as file:
        for img in completed_images:
            file.write(f"{img}\n")


def generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range):
    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    progress_path = os.path.join(weights_save_path, "epsilon20_0.9.txt")
    completed_images = load_completed_images(progress_path)
    expansion_counts = {}
    weights_A = {}
    weights_B = {}

    for filename in os.listdir(original_images_folder):
        if filename.endswith(".png") and filename not in completed_images:
            logging.info(f"Starting to process {filename}")
            original_image_path = os.path.join(original_images_folder, filename)
            original_image = Image.open(original_image_path)
            h, w, original_img_array = linearize_pixels(original_image)
            original_class = test_classifier(h, w, original_img_array.flatten())

            belongs_to_A, ratio = generate_images_and_weights_with_normal_distribution(h, w, original_img_array, epsilon_range, original_class, images_save_path, filename, expansion_counts)

            if belongs_to_A:
                weights_A[filename] = ratio
            else:
                weights_B[filename] = 0  # Initially, set all weights for B to 0

            update_completed_images(progress_path, filename, completed_images)

    # Calculate weights for B
    total_weight_B = 0.1
    number_of_images_B = len(weights_B)
    weight_per_image_B = total_weight_B / number_of_images_B if number_of_images_B else 0
    for image in weights_B:
        weights_B[image] = weight_per_image_B

    # Combine weights A and B
    weights = {**weights_A, **weights_B}

    # Save weights to JSON file
    weights_json_path = os.path.join(weights_save_path, "weights20_0.9.json")
    with open(weights_json_path, "w") as json_file:
        json.dump(weights, json_file)

    logging.info(f"Weights saved to {weights_json_path}")


# def generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range):
#     if not os.path.exists(images_save_path):
#         os.makedirs(images_save_path)
#     if not os.path.exists(weights_save_path):
#         os.makedirs(weights_save_path)
#
#     progress_path = os.path.join(weights_save_path, "epsilon20_0.6.txt")
#     completed_images = load_completed_images(progress_path)
#     expansion_counts = {}
#     weights_A = {}
#     total_s = 0  # 用于累计集合A中所有图片的分数之和
#
#     for filename in os.listdir(original_images_folder):
#         if filename.endswith(".png") and filename not in completed_images:
#             logging.info(f"Starting to process {filename}")
#             original_image_path = os.path.join(original_images_folder, filename)
#             original_image = Image.open(original_image_path)
#             h, w, original_img_array = linearize_pixels(original_image)
#             original_class = test_classifier(h, w, original_img_array.flatten())
#
#             belongs_to_A, ratio = generate_images_and_weights_with_normal_distribution(h, w, original_img_array, epsilon_range, original_class, images_save_path, filename, expansion_counts)
#
#             if belongs_to_A:
#                 weights_A[filename] = ratio
#                 total_s += ratio  # 累计所有属于集合A的图片的分数
#
#             update_completed_images(progress_path, filename, completed_images)
#
#     # 计算集合A中每张图片的权重
#     total_weight_A = 0.6  # 集合A的总权重分配
#     for filename, ratio in weights_A.items():
#         weights_A[filename] = (ratio / total_s) * total_weight_A
#
#     # 计算并分配集合B的权重
#     weights_B = {}
#     total_weight_B = 0.4  # 集合B的总权重分配
#     number_of_images_B = len(completed_images) - len(weights_A)  # 假设所有处理过的图片减去集合A的图片数为集合B的图片数
#     weight_per_image_B = total_weight_B / number_of_images_B if number_of_images_B else 0
#     for filename in completed_images:
#         if filename not in weights_A:  # 如果图片不在集合A中，则它属于集合B
#             weights_B[filename] = weight_per_image_B
#
#     # 合并集合A和集合B的权重，并保存到JSON文件
#     weights = {**weights_A, **weights_B}
#     weights_json_path = os.path.join(weights_save_path, "weights20_0.6.json")
#     with open(weights_json_path, "w") as json_file:
#         json.dump(weights, json_file)
#
#     logging.info(f"Weights saved to {weights_json_path}")



# 这里是调用函数的地方，你需要根据你的实际情况替换路径

original_images_folder = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/DBImages"
images_save_path = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/epsilonExpandOptimization"
weights_save_path = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/epsilonExpandWeights"
epsilon_range = range(1, 21)  # 示例ε值范围
check_path_and_permissions(images_save_path)
check_path_and_permissions(weights_save_path)

generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range)