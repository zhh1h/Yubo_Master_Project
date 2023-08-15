import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from t_ngd_cifar10 import test_classifier
from t_ngd_cifar10 import linearize_pixels
from t_ngd_cifar10 import create_f
from t_ngd_cifar10 import classes
from t_ngd_cifar10 import net
import os
import argparse
import shap
import torch


parser = argparse.ArgumentParser(description='SHAP')

parser.add_argument('--target', type=str, help='Target class', required=False)
args = parser.parse_args()

net = net.cuda()


def load_images_from_folder(folder_path_background):
    image_files = sorted(os.listdir(folder_path_background))
    images = []
    for img_file in image_files:
        img_path = os.path.join(folder_path_background, img_file)
        img = Image.open(img_path)
        h, w, img_array = linearize_pixels(img)
        img_tensor = torch.Tensor(img_array).view(1,3,32,32).cuda()
        images.append(img_tensor)
    return torch.cat(images,0)


folder_path_background = './data/airplane'
background = torch.stack(load_images_from_folder(folder_path_background))
background = background.cuda()

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

        #Convert image pixel values to tuple of tuples format
        pixel_tuples = tuple(tuple(map(int, img[i, j] * 255)) for i in range(32) for j in range(32))

        # Print the pixel values of the generated image with its index
        print(f"Image {idx+1} Pixels: {pixel_tuples}")

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



def compute_shap_values(model, background, input_data):
    print(f"background's shape{background.shape}")
    print(f"input_data's shape{input_data.shape}")
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(input_data)
    #print(input_data.shape)  # 应该打印出 [1, channels, height, width]
    return shap_values

def adjust_image_based_on_shap(original_img, shap_values, target_class_index, intensity=0.1):
    shap_for_target = shap_values[target_class_index]
    adjusted_img = original_img + intensity * shap_for_target
    return adjusted_img


def calculate_f_scores_in_folder_with_shap(folder_path, target_class=None,threshold = 0.5):
    selected_images = []
    image_filenames = sorted(os.listdir(folder_path))

    for index, filename in enumerate(image_filenames):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        h, w, img_array = linearize_pixels(img)
        identified_class = test_classifier(h, w, img_array)
        img_tensor = torch.Tensor(img_array).view(1,3,32,32).cuda()

        if target_class is not None:
            f_function = create_f(h, w, target_class)
            f_score = f_function(img_array)
            print(f"Image {filename}: f score {f_score}, class is {classes[identified_class]}")

        #using model to calculate shap values:
        shap_values = compute_shap_values(net,background,img_tensor.unsqueeze(0))

        # adjust image based on shap values
        adjusted_img_array = adjust_image_based_on_shap(img_array,shap_values,target_class)

        # using new image after adjusting to evaluate:

        if target_class is not None:
            f_function_after = create_f(h, w, target_class)
            f_score_after = f_function_after(adjusted_img_array)
            print(f"Image {filename}: f score {f_score_after}, class is {identified_class}")

            if f_score_after > threshold:
                selected_images.append((index,filename,f_score_after))

    return selected_images




#folder_path = "./shap_random_images"
folder_path = os.path.join(os.path.dirname(__file__), "shap_random_images")

generate_and_save_images(num_seeds,folder_path)


selected_images = calculate_f_scores_in_folder_with_shap(folder_path, target_class=classes.index(args.target),threshold = 0.5)

#输出筛选结果
print("Selected images:")
for index,filename,f_score in selected_images:
    print(f"Original Index:{index},Image:{filename},f Score:{f_score_after}")




