import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from t_ngd_cifar10 import test_classifier
from t_ngd_cifar10 import linearize_pixels
from t_ngd_cifar10 import create_f
from t_ngd_cifar10 import classes
import os
import argparse

parser = argparse.ArgumentParser(description='SHAP')

parser.add_argument('--target', type=str, help='Target class', required=False)
args = parser.parse_args()

def generate_random_image(seed, shape=(32, 32, 3)):
    np.random.seed(seed)
    return np.random.rand(*shape)

num_seeds = 10
random_images = [generate_random_image(seed) for seed in range(num_seeds)]

# 生成随机图片并保存到文件夹
def generate_and_save_images(num_images, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx in range(num_images):
        img = generate_random_image(idx)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_filename = f"random_image_seed_{idx}.jpg"
        img_path = os.path.join(folder_path, img_filename)
        img_pil.save(img_path)

# 遍历文件夹中的图片并显示
def display_images_in_folder(folder_path):
    image_filenames = os.listdir(folder_path)

    for idx, filename in enumerate(image_filenames):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Seed: {idx}")
        plt.show()


def calculate_f_scores_in_folder(folder_path, target_class=None,threshold = 0.5):
    selected_images = []
    image_filenames = sorted(os.listdir(folder_path))

    for index, filename in enumerate(image_filenames):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        h, w, img_array = linearize_pixels(img)

        if target_class is not None:
            f = create_f(h, w, target_class)
            print(f"Image {filename}: f score {f}")

            if f > threshold:
                selected_images.append((index,filename,f))

    return selected_images

#folder_path = "./shap_random_images"
folder_path = os.path.join(os.path.dirname(__file__), "shap_random_images")


selected_images = calculate_f_scores_in_folder(folder_path, target_class=classes.index(args.target),threshold = 0.5)

#输出筛选结果
print("Selected images:")
for index,filename,f_score in selected_images:
    print(f"Original Index:{index},Image:{filename},f Score:{f_score}")




