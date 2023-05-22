from PIL import Image
import random
import os

# 定义要生成的图片数量和保存的文件夹
num_images =  10000
save_dir = '../../data/random_images_random/'

# 创建保存文件夹（如果不
# 存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
low_index = 3
high_index = 3
l_index = []
h_index = []
for il in range(low_index):
    l = random.randint(0,127)
    l_index.append(l)
for ih in range(high_index):
    h = random.randint(128,255)
    h_index.append(h)

print(l_index)
print(h_index)




# 循环生成指定数量的随机图像
for i in range(num_images):
    # random generate the 200x200 RGB images
    img = Image.new('RGB', (200, 200), (255, 255, 255))

    # filling the random pixel value
    pixels = img.load()
    for j in range(img.size[0]):
        for k in range(img.size[1]):
            r = random.randint(l_index[0], h_index[0])
            g = random.randint(l_index[1], h_index[1])
            b = random.randint(l_index[2], h_index[2])
            pixels[j, k] = (r, g, b)


    imgSavePath = os.path.join(save_dir, f'random_image{i}.jpg')
    img.save(imgSavePath)
    print(f'img; {i} have been saved to {imgSavePath}')



num_images =  10000
save_dir = '../../data/random_images_random/'

# 创建保存文件夹（如果不
# 存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)





# 循环生成指定数量的随机图像
for i in range(num_images):
    # random generate the 200x200 RGB images
    img = Image.new('RGB', (200, 200), (255, 255, 255))

    # filling the random pixel value
    pixels = img.load()
    for j in range(img.size[0]):
        for k in range(img.size[1]):
            r = random.randint(0, 127)
            g = random.randint(0, 127)
            b = random.randint(0, 127)
            pixels[j, k] = (r, g, b)


    imgSavePath = os.path.join(save_dir, f'random_image_0_127{i}.jpg')
    img.save(imgSavePath)
    print(f'img; {i} have been saved to {imgSavePath}')


num_images =  10000
save_dir = '../../data/random_images_random/'

# 创建保存文件夹（如果不
# 存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)





# 循环生成指定数量的随机图像
for i in range(num_images):
    # random generate the 200x200 RGB images
    img = Image.new('RGB', (200, 200), (255, 255, 255))

    # filling the random pixel value
    pixels = img.load()
    for j in range(img.size[0]):
        for k in range(img.size[1]):
            r = random.randint(128, 255)
            g = random.randint(128, 255)
            b = random.randint(128, 255)
            pixels[j, k] = (r, g, b)


    imgSavePath = os.path.join(save_dir, f'random_image_128_255{i}.jpg')
    img.save(imgSavePath)
    print(f'img; {i} have been saved to {imgSavePath}')


