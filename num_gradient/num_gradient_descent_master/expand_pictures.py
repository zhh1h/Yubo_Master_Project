from PIL import Image
import numpy as np
from t_ngd_cifar10 import test_classifier, linearize_pixels
#from t_ngd_cifar10 import net
from t_ngd_cifar10 import preprocess_image


# 假设 test_classifier 函数已经定义，这里仅为示例



# 修改后的 generate_image_with_global_noise 函数，添加了保存图片和分类功能
def generate_image_with_noise_and_classify(h, w, img_array, std_deviation):
    original_image = img_array.reshape((h, w, 3)).astype('uint8')
    noise = np.random.normal(0, std_deviation, original_image.shape)
    new_image = original_image + noise
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)

    # 使用预处理函数转换新图像为张量
    img_tensor = preprocess_image(h, w, new_image)

    # 保存新图像
    img_path = f"new_Image{std_deviation}.jpg"
    Image.fromarray(new_image, 'RGB').save(img_path)
    print(f"图像保存在 {img_path}")

    # 使用 test_classifier 函数进行分类
    predicted_class = test_classifier(h, w, img_tensor)
    print(f"新图像的预测类别：{predicted_class}")


your_original_image = Image.open("./output.jpg")
h, w, img_array = linearize_pixels(your_original_image)
std_deviation = 0
generate_image_with_noise_and_classify(h, w, img_array, std_deviation)