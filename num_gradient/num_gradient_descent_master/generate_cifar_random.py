
from PIL import Image
import random
import os

# 定义要生成的图片数量和保存的文件夹
num_images = 1
save_dir = './data/generate_cifar_random/'

# 创建保存文件夹（如果不
# 存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 循环生成指定数量的随机图像
for i in range(num_images):
    # random generate the 32by32 RGB images
    img = Image.new('RGB', (32, 32), (127, 127, 127))

    # filling the random pixel value
    pixels = img.load()
    for j in range(img.size[0]):
        for k in range(img.size[1]):
            #r = random.randint(0,127)
            r = 127
            g = 127
            b = 127
            #g = random.randint(0,127)
            #b = random.randint(0,127)
            pixels[j, k] = (r, g, b)


    imgSavePath = os.path.join(save_dir, f'random_image_127{i}.jpg')
    img.save(imgSavePath)
    print(f'img; {i} have been saved to {imgSavePath}')



