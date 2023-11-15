from PIL import Image
import numpy as np
import os
import torch
import sys

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")
from cifar10vgg19testClassifier import test_classifier, linearize_pixels
from models import *

SAVE_PATH = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/truck0'

def predictCaltech(h, w, img_array):
    return test_classifier(h, w, img_array, return_confidence=True)

folder_path = ('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/truck/valid/Truck')
high_confidence_images = []

for img_file in os.listdir(folder_path):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, img_file)
        your_original_image = Image.open(img_path).convert("RGB")
        h, w, img_array = linearize_pixels(your_original_image)

        predicted_class, confidence = predictCaltech(h, w, img_array)

        if confidence > 0.7 and predicted_class == "truck":
            img_name = os.path.splitext(img_file)[0]
            print(f"Image name: {img_name}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence}")

            save_path = os.path.join(SAVE_PATH, f"{img_name}.png")
            original_image = img_array.reshape((h, w, 3)).astype('uint8')
            processed_img = Image.fromarray(original_image, 'RGB')
            processed_img.save(save_path)

            high_confidence_images.append({
                "image_name": img_name,
                "predicted_class": predicted_class,
                "confidence": confidence
            })

print("\nSummary of images with confidence greater than 0.7:")
for item in high_confidence_images:
    print(f"Image Name: {item['image_name']}, Predicted Class: {item['predicted_class']}, Confidence: {item['confidence']}")
