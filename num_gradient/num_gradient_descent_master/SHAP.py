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
background_data= load_images_from_folder(folder_path_background)
background = background_data.cuda()

# 打印background的形状
#print("Shape of background:", background.shape)

# def generate_random_image(seed, shape=(32, 32, 3)):
#     np.random.seed(seed)
#     return np.random.rand(*shape)
#
num_seeds = 10
# random_images = [generate_random_image(seed) for seed in range(num_seeds)]

def compute_shap_values_for_background(model,background):
    explainer = shap.DeepExplainer(model,background)
    shap_values = explainer.shap_values(background)
    return shap_values

def generate_image_based_on_shap(shap_values,seed,shap = (3,32,32)):
    np.random.seed(seed)
    random_multiplier = np.random.rand(*shap)
    image = random_multiplier * shap_values
    image = np.clip(image,0,255)
    return image

def save_generated_images(shap_values,num_images,folder_path):
    if not os.path.exits(folder_path):
        os.makedirs(folder_path)
    for idx in range(num_images):
        random_image = generate_image_based_on_shap(shap_values,idx)
        img_pil = Image.fromarry(random_image.astype(np.unit8))
        img_filename = f'random_image_seed{idx}.jpg'
        img_path = os.path.join(folder_path,img_filename)
        img_pil.save(img_path)



# 生成随机图片并保存到文件夹
# def generate_random_image_from_shap(explainer,target_class_index,intensity = 2,shap = (32,32,3)):
#     baseline = np.mean(background_data.cpu().numpy(),axis = 0)
#     shap_values = explainer.shap_values(torch.zeros(1,3,32,32).cuda())
#     shap_for_target = shap_values[target_class_index]
#     random_image = baseline +intensity * shap_for_target
#     random_image = np.clip(random_image,0,255)
#     return random_image



#folder_path = "./shap_random_images"
# # 遍历文件夹中的图片并显示
# def display_images_in_folder(folder_path):
#     image_filenames = os.listdir(folder_path)
#
#     for idx, filename in enumerate(image_filenames):
#         image_path = os.path.join(folder_path, filename)
#         img = Image.open(image_path)
#         plt.imshow(img)
#         plt.title(f"Seed: {idx}")
#         plt.show()


#
# def compute_shap_values(model, background, input_data):
#     #print(f"background's shape{background.shape}")
#     #print(f"input_data's shape{input_data.shape}")
#     explainer = shap.DeepExplainer(model, background)
#     shap_values = explainer.shap_values(input_data)
#     #print(input_data.shape)  # 应该打印出 [1, channels, height, width]
#     return shap_values

# def generate_and_save_images_from_shap(num_images,explainer,target_class_index,folder_path):
#     if not os.path.exits(folder_path):
#         os.makedirs(folder_path)
#     for idx in range(num_images):
#         random_image = generate_random_image_from_shape(explainer,target_class_index)
#         img_pil = Image.fromarry(random_image.astype(np.unit8))
#         img_filename = f'random_image_seed{idx}.jpg'
#         img_path = os.path.join(folder_path,img_filename)
#         img_pil.save(img_path)
#
# def adjust_image_based_on_shap(original_img, shap_values, target_class_index, intensity=2):
#     # reshape the original_img to (3,32,32)
#     reshaped_img = original_img.reshape(3, 32, 32)
#     shap_for_target = shap_values[target_class_index]
#     adjusted_img = reshaped_img + intensity * shap_for_target
#     adjusted_img = np.clip(adjusted_img,0,255)
#     return adjusted_img


def calculate_f_scores_in_folder_with_shap(folder_path, target_class=None,threshold = 0.5):
    selected_images = []
    image_filenames = sorted(os.listdir(folder_path))

    for index, filename in enumerate(image_filenames):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        h, w, img_array = linearize_pixels(img)
        identified_class = test_classifier(h, w, img_array)
        img_tensor = torch.Tensor(img_array).view(3,32,32).cuda()
        # 打印img_tensor增加维度后的形状
        #print("Shape of img_tensor after unsqueeze:", img_tensor.unsqueeze(0).shape)

        if target_class is not None:
            f_function = create_f(h, w, target_class)
            f_score = f_function(img_array)
            print(f"Image {filename}: f score on target class{classes.index(args.target)} is {f_score}, now class is {identified_class}{classes[identified_class]}")

        #using model to calculate shap values:
        #shap_values = compute_shap_values(net,background,img_tensor.unsqueeze(0))

        #print(f"SHAP values for Image {filename}: {shap_values}")

        # adjust image based on shap values
        #adjusted_img_array = adjust_image_based_on_shap(img_array,shap_values,target_class)

        # using new image after adjusting to evaluate:

        # if target_class is not None:
        #     f_function_after = create_f(h, w, target_class)
        #     f_score_after = f_function_after(adjusted_img_array)
        #     print(f"Image {filename}: f_score_after on target class{classes.index(args.target)} is {f_score_after},  now class is {identified_class}{classes[identified_class]}")

        if f_score > threshold:
            selected_images.append((index,filename,f_score))

    return selected_images




# 计算背景数据集的SHAP值
shap_values_background = compute_shap_values_for_background(net, background)
# 取平均SHAP值
avg_shap_values = np.mean(shap_values_background, axis=0)

folder_path = "./shap_generated_images"
save_generated_images(avg_shap_values, 10, folder_path)

# 如果有指定目标类别，则使用它，否则默认为0
target_class = classes.index(args.target) if args.target else 0
selected_images = calculate_f_scores_in_folder(folder_path, target_class=target_class, threshold=0.5)

# 输出筛选结果
print("Selected images:")
for index, filename, f_score in selected_images:
    print(f"Original Index: {index}, Image: {filename}, f Score: {f_score}")
    ######
