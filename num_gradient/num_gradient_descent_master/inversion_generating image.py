from PIL import Image
import random
import os
num_images =  10
save_dir = '../../data/num_random_images/'

# 创建保存文件夹（如果不
# 存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)





# 循环生成指定数量的随机图像
for i in range(num_images):
    # random generate the 32*32 RGB images
    img = Image.new('RGB', (32, 32), (255, 255, 255))

    # filling the random pixel value
    pixels = img.load()
    for j in range(img.size[0]):
        for k in range(img.size[1]):
            r = random.randint(128, 255)
            g = random.randint(128, 255)
            b = random.randint(128, 255)
            pixels[j, k] = (r, g, b)


    imgSavePath = os.path.join(save_dir, f'random_image{i}.jpg')
    img.save(imgSavePath)
    print(f'img; {i} have been saved to {imgSavePath}')


