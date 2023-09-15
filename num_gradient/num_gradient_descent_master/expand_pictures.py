from PIL import Image
import numpy as np
from t_ngd_cifar10 import test_classifier, linearize_pixels
import  os
import torch
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

if not os.path.exists('std_deviation'):
    os.makedirs('std_deviation')


# 修改后的 generate_image_with_global_noise 函数，添加了保存图片和分类功能
def generate_image_with_noise_and_classify(h, w, img_array, std_deviation):
    original_image = img_array.reshape((h, w, 3)).astype('uint8')

    # 生成高斯噪声并添加到原图
    noise = np.random.normal(0, std_deviation, original_image.shape)
    new_image = original_image + noise
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    # 保存新图像
    # img_path = f"new_Image{std_deviation}{predicted_class}.png"
    # Image.fromarray(new_image, 'RGB').save(img_path)

    # 对新图像进行分类
    h, w, img_array = linearize_pixels(new_image)
    predicted_class = test_classifier(h, w, img_array)
    print(f"new image class：{predicted_class}")

    img_path = f"std_deviation/new_Image_138_{std_deviation}_{predicted_class}.png"
    Image.fromarray(new_image, 'RGB').save(img_path)
    return predicted_class


# 使用 linearize_pixels 函数处理原始图像，并得到高度 h，宽度 w，和一维数组 img_array
your_original_image = Image.open("./output.png")  # 这里使用您自己的图像路径
h, w, img_array = linearize_pixels(your_original_image)

# 获取原图像的分类
original_class = test_classifier(h, w, img_array)
print(f"original image class：{original_class}")

# 初始化标准差
std_deviation = 0

# 循环以找出导致分类改变的最小标准差
while True:
    print(f"test standard deviation：{std_deviation}")
    new_class = generate_image_with_noise_and_classify(h, w, img_array, std_deviation)

    if new_class != original_class:
        print(f"When the standard deviation is {std_deviation}, the class changes to {new_class}")
        break  # 结束循环

    std_deviation += 10 # 或者您也可以选择其他的增加幅度，比如 += 5，根据实际情况调整