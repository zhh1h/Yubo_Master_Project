import os
import numpy as np
from PIL import Image
import json
import sys
import logging

sys.path.append(
    "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

# 配置日志记录
logging.basicConfig(
    filename='/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/image_generation.log',
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Program started.")


def generate_images_and_weights_with_normal_distribution(h, w, original_img_array, epsilon_range, original_class,
                                                         images_save_path, filename, expansion_counts):
    different_class_count = 0
    for epsilon in epsilon_range:
        try:
            logging.info(f"Generating image for {filename} with epsilon {epsilon}")
            covariance_matrix = np.diag([epsilon, epsilon, epsilon])
            noise = np.random.multivariate_normal([0, 0, 0], covariance_matrix, (h, w))
            new_image_array = original_img_array.reshape((h, w, 3)) + noise
            new_image_array = np.clip(new_image_array, 0, 255).astype(np.uint8)

            new_image = Image.fromarray(new_image_array, 'RGB')
            new_image_path = os.path.join(images_save_path, f"{os.path.splitext(filename)[0]}_epsilon_{epsilon}.png")
            new_image.save(new_image_path)

            expansion_counts[filename] = expansion_counts.get(filename, 0) + 1
            logging.info(f"Successfully saved: {new_image_path}")

            predicted_class = test_classifier(h, w, new_image_array.flatten())
            if predicted_class != original_class:
                different_class_count += 1
        except Exception as e:
            logging.error(f"Exception for {filename} with epsilon {epsilon}: {str(e)}")

    if different_class_count > 0:
        weight = different_class_count / len(epsilon_range)
        return {filename: weight}
    else:
        return {}


def load_completed_images(progress_path):
    completed_images = set()
    try:
        with open(progress_path, "r") as file:
            completed_images = set(file.read().splitlines())
    except FileNotFoundError:
        pass
    return completed_images


def update_completed_images(progress_path, filename, completed_images):
    completed_images.add(filename)
    with open(progress_path, "w") as file:
        file.writelines(f"{name}\n" for name in completed_images)


def generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range):
    print(f"Total original images: {len(os.listdir(original_images_folder))}")
    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
    progress_path = os.path.join(weights_save_path, "progress.txt")
    completed_images = load_completed_images(progress_path)

    expansion_counts = {}
    weights_record = {}

    for filename in os.listdir(original_images_folder):
        if filename.endswith(".png") and filename not in completed_images:
            logging.info(f"Starting to process {filename}")
            original_image_path = os.path.join(original_images_folder, filename)
            original_image = Image.open(original_image_path)
            h, w, original_img_array = linearize_pixels(original_image)
            original_class = test_classifier(h, w, original_img_array.flatten())

            weight_info = generate_images_and_weights_with_normal_distribution(h, w, original_img_array, epsilon_range,
                                                                               original_class, images_save_path,
                                                                               filename, expansion_counts)
            if weight_info:
                weights_record.update(weight_info)

            update_completed_images(progress_path, filename, completed_images)

    print(f"Expansion counts: {expansion_counts}")
    weights_json_path = os.path.join(weights_save_path, "weights_record.json")
    with open(weights_json_path, "w") as json_file:
        json.dump(weights_record, json_file)

    print(f"All weights saved to {weights_json_path}")


# 调用函数的部分
original_images_folder = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/DBImages"
images_save_path = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/epsilonExpand"
weights_save_path = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/epsilonExpandWeights"
epsilon_range = range(1, 21)
completed_images = load_completed_images(os.path.join(weights_save_path, "progress.txt"))
generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range)

