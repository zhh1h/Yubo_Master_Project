import os
import numpy as np
from PIL import Image
import json
import sys
import logging

sys.path.append(
    "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/Caltech256TrainandPredict")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels

torch.backends.cudnn.enabled = False

print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 配置日志记录
logging.basicConfig(
    filename='/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/image_generation.log',
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Program started.")


def expand_and_classify(filename, epsilon_range, original_img_array, original_class, images_save_path):
    different_class_count = 0  # n
    total_expansions = len(epsilon_range)  # N
    for epsilon in epsilon_range:
        # Generate and save expansion images, similar logic as previously described
        # Assuming existing logic for image expansion and classification
        # For simplicity, let's say we just increment different_class_count if class differs
        predicted_class = "Some logic to predict class"  # Placeholder for actual classification logic
        if predicted_class != original_class:
            different_class_count += 1
    # Return the ratio if there's at least one image of a different class, else None
    if different_class_count > 0:
        return different_class_count / total_expansions
    else:
        return None


def generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range):
    set_A = {}
    set_B = []
    for filename in os.listdir(original_images_folder):
        # Load the original image and classify
        original_image_path = os.path.join(original_images_folder, filename)
        original_image = Image.open(original_image_path)
        h, w, original_img_array = linearize_pixels(original_image)
        original_class = test_classifier(h, w, original_img_array.flatten())  # Assuming this returns the class

        ratio = expand_and_classify(filename, epsilon_range, original_img_array, original_class, images_save_path)
        if ratio is not None:
            set_A[filename] = ratio
        else:
            set_B.append(filename)

    # Calculate weights
    weights = calculate_weights(set_A, set_B)

    # Save weights to JSON
    weights_json_path = os.path.join(weights_save_path, "image_weights.json")
    with open(weights_json_path, "w") as json_file:
        json.dump(weights, json_file)
    print(f"Weights saved to {weights_json_path}")


def calculate_weights(set_A, set_B):
    total_weight_A = 0.5
    total_weight_B = 0.5
    weights = {}

    # Weights for set B
    weight_per_image_B = total_weight_B / len(set_B) if set_B else 0
    for image in set_B:
        weights[image] = weight_per_image_B

    # Weights for set A
    total_s = sum(set_A.values())
    for image, s in set_A.items():
        weights[image] = (s / total_s) * total_weight_A if total_s else 0

    return weights


# Assuming the existence of functions like test_classifier and linearize_pixels
# Replace the following paths with your actual paths
original_images_folder = "/path/to/your/original_images_folder"
images_save_path = "/path/to/your/images_save_path"
weights_save_path = "/path/to/your/weights_save_path"
epsilon_range = range(1, 21)  # Example range

generate_images_and_weights_for_folder(original_images_folder, images_save_path, weights_save_path, epsilon_range)
