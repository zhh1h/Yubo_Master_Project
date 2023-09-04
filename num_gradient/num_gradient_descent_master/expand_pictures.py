from PIL import Image
import numpy as np
from t_ngd_cifar10 import test_classifier, linearize_pixels
from t_ngd_cifar10 import net


# 假设 test_classifier 函数已经定义，这里仅为示例



# 修改后的 generate_image_with_global_noise 函数，添加了保存图片和分类功能
def generate_image_with_noise_and_classify(h, w, img_array, std_deviation):
    # 重新将一维数组转换为多维数组（图像）
    original_image = img_array.reshape((h, w, 3)).astype('uint8')

    # 生成与原图相同形状的随机噪声（这里没有使用协方差矩阵）
    noise = np.random.normal(0, std_deviation, original_image.shape)

    # 将随机噪声添加到原图上
    new_image = original_image + noise

    # 限制新像素值在合法范围内（例如，0到255对于uint8图像）
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    #net.eval()
    # 保存新图像
    img_path = f"new_Image{std_deviation}.jpg"
    Image.fromarray(new_image, 'RGB').save(img_path)
    print(f"图像保存在 {img_path}")

    # 使用 test_classifier 函数进行分类
    img_array = new_image.reshape(h * w * 3).astype('float64')
    predicted_class = test_classifier(h, w, img_array)
    print(f"新图像的预测类别：{predicted_class}")

# 使用 linearize_pixels 函数处理原始图像，并得到高度 h，宽度 w，和一维数组 img_array
your_original_image = Image.open("./output.jpg")
h, w, img_array = linearize_pixels(your_original_image)

# 使用 generate_image_with_noise_and_classify 函数添加噪声并分类
std_deviation = 1  # 这个值可以根据需要进行调整
generate_image_with_noise_and_classify(h, w, img_array, std_deviation)