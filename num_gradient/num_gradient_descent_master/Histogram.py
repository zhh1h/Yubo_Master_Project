import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='CIFAR10 Security Attacks')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
#parser.add_argument('--target', type=str, help='Target class', required=False)
args = parser.parse_args()
# 加载原始图像

if args.input_pic:
        #print("There is input pic")
        #img = Image.open(args.input_pic)
        image = Image.open(args.input_pic)
        image = np.array(image)


# 生成一张随机噪声图片
noise_image = np.random.randint(0, 256, image.shape, dtype=np.uint8)

# 这里需要对每个颜色通道分别进行直方图匹配
matched_image = np.zeros_like(noise_image)
for channel in range(3):
    hist_image, bins_image = np.histogram(image[..., channel].flatten(), bins=256, range=[0,256])
    hist_noise_image, bins_noise_image = np.histogram(noise_image[..., channel].flatten(), bins=256, range=[0,256])
    cdf_image = hist_image.cumsum()
    cdf_noise_image = hist_noise_image.cumsum()
    cdf_image_normalized = cdf_image * hist_noise_image.max() / cdf_image.max()
    cdf_noise_image_normalized = cdf_noise_image * hist_noise_image.max() / cdf_noise_image.max()
    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while cdf_noise_image_normalized[j] < cdf_image_normalized[i] and j < 255:
            j += 1
        lut[i] = j
    matched_image[..., channel] = lut[noise_image[..., channel]]

# 显示原始图像、随机噪声图片和匹配后的图片
Image.fromarray(image).save('original_image_1.jpg')
Image.fromarray(noise_image).save('noise_image.jpg')
Image.fromarray(matched_image).save('matched_image_1.jpg')
