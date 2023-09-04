from PIL import Image
import numpy as np
from t_ngd_cifar10 import test_classifier, linearize_pixels


# 假设 test_classifier 函数已经定义，这里仅为示例



# 修改后的 generate_image_with_global_noise 函数，添加了保存图片和分类功能
def generate_image_with_global_noise_and_classify(h, w, img_array, std_deviation):
    # 将1D数组重新整形为多维数组（图像）
    original_image = img_array.reshape((h, w, 3)).astype('uint8')

    # 为简单起见，我们将使用单位矩阵作为协方差矩阵
    identity_matrix = np.eye(h * w * 3)

    # 根据协方差矩阵生成随机噪声
    noise = np.random.multivariate_normal(np.zeros(h * w * 3), identity_matrix * std_deviation ** 2)
    noise = noise.reshape((h, w, 3))

    # 将随机噪声添加到原图像上
    new_image = original_image + noise.astype('uint8')

    # 将新像素值限制在合法范围内（例如，对于uint8图像是0到255）
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    # 保存新图像
    img_path = f"new_Image{std_deviation}.jpg"
    Image.fromarray(new_image, 'RGB').save(img_path)
    print(f"图片已保存在 {img_path}")

    # 使用test_classifier函数进行分类
    img_array = new_image.reshape(h * w * 3).astype('float64')
    predicted_class = test_classifier(h, w, img_array)
    print(f"新图像的预测类别是：{predicted_class}")


# 使用linearize_pixels函数处理原始图像，得到高度h、宽度w和1D数组img_array
your_original_image = Image.open("./output.jpg")  # 用您图像的路径替换
h, w, img_array = linearize_pixels(your_original_image)

# 使用generate_image_with_global_noise_and_classify函数添加噪声并分类
std_deviation = 0  # 标准差（或“扩展”）可以根据需要进行调整
generate_image_with_global_noise_and_classify(h, w, img_array, std_deviation)