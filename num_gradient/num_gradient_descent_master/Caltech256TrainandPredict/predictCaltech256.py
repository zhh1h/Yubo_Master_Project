from PIL import Image
import numpy as np
import os
import torch
import sys

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")
from t_ngd_cifar10 import test_classifier, linearize_pixels
from models import *

SAVE_PATH = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/truck0'
def predictCaltech(h, w, img_array):
    original_image = img_array.reshape((h, w, 3)).astype('float64')
    h, w, img_array = linearize_pixels(original_image)
    predicted_class, confidence = test_classifier(h, w, img_array, return_confidence=True)

    result = {
        "image_name": os.path.splitext(os.path.basename(img_path))[0],
        "predicted_class": predicted_class,
        "confidence": confidence
    }

    if confidence > 0.85 and predicted_class == "truck":
        print(f"Image name: {result['image_name']}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {confidence}")

        save_path = os.path.join(SAVE_PATH, f"{result['image_name']}.png")
        processed_img = Image.fromarray(original_image.astype('uint8'), 'RGB')
        processed_img.save(save_path)
    print("--------------")

    return result


folder_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/truck/valid/Truck'
high_confidence_images = []

for img_file in os.listdir(folder_path):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, img_file)
        your_original_image = Image.open(img_path).convert("RGB")
        h, w, img_array = linearize_pixels(your_original_image)

        result = predictCaltech(h, w, img_array)

        if result['confidence'] > 0.85 and result['predicted_class'] == "truck":
            high_confidence_images.append(result)

print("\nSummary of images with confidence greater than 0.85:")
for item in high_confidence_images:
    print(
        f"Image Name: {item['image_name']}, Predicted Class: {item['predicted_class']}, Confidence: {item['confidence']}")


