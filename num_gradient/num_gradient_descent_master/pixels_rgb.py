from PIL import Image
from t_ngd_cifar10 import test_classifier
from t_ngd_cifar10 import linearize_pixels
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


parser = argparse.ArgumentParser(description='CIFAR10 Security Attacks')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
parser.add_argument('--target', type=str, help='Target class', required=False)
args = parser.parse_args()
def pixel_rgb(a):

    im = Image.open(a)

    for y in range(im.size[1]):
        for x in range(im.size[0]):
            pix = im.getpixel((x,y))
            print(pix)
    return pix

if args.input_pic:
        #print("There is input pic")
        img = Image.open(args.input_pic)
        h, w, img_array = linearize_pixels(img)
        test_classifier(h,w,img_array)


#pixel_rgb(args.input_pic)

# 提取像素值

pixels = img_array.flatten()
print(pixels)






# 绘制直方图
plt.hist(pixels, bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Pixel Value Distribution')
plt.show()

mean = np.mean(pixels)
variance = np.var(pixels)
std_dev = np.std(pixels)
print(f"mean:{mean}")
print(f"variance:{variance}")
print(f"std_deviation:{std_dev}")
size = len(pixels)
print(size)

#arr_generate = np.random.normal(mean, std_dev, size=size)
low = 0 # 数字范围的下界
high = 255  # 数字范围的上界

kde = gaussian_kde(pixels)
arr_generate = kde.resample(len(pixels))[0]
arr_generate = np.clip(arr_generate, low, high)
# arr_generate = arr_generate.astype(np.uint8)

print(arr_generate)
#trasfer to 3 channel
#rgb_arr = arr_generate.reshape((3, -1))
#plt.imshow(rgb_arr.transpose(1, 0, 2))

# 转换为三通道的数组
arr_generate = arr_generate.reshape((32, 32, 3))
arr_generate = arr_generate.astype(np.uint8)

# 显示图片

plt.imshow(arr_generate)
plt.axis('off')
plt.savefig('distribute_img2.jpg',format='jpg')
plt.show()


# b = 'distribute_img2.jpg'

#generate_img = pixel_rgb(b)

pixels_arr = arr_generate.flatten()
print(pixels_arr)

