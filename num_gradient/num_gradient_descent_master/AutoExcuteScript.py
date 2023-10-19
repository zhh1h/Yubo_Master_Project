from __future__ import print_function
import os
import sys
#sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/knockoff/models")
#
import os

# 定义目标类别
targets = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship"]

# 在循环外部，初始化上一次的图片编号
previous_i = 42

# 开始循环
for i in range(43, 128):  # 从fix_pixels241到fix_pixels256

    # 在循环开始时，替换上一次的图片编号
    with open("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/expand_pictures.py", "r") as file:
        contents = file.read()
        contents = contents.replace(f'new_Image_{previous_i}_', f'new_Image_{i}_')

    with open("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/expand_pictures.py", "w") as file:
        file.write(contents)

    image_path = f"/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/fix_pixels_images/fix_pixels{i}.jpg"

    # 对于每一张图片，进行9次模型反转攻击
    for target in targets:
        # 运行程序A
        os.system(f"{sys.executable} /home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/t_ngd_cifar10.py -i {image_path} --target {target}")

        # 运行程序B
        os.system(f"{sys.executable} /home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/expand_pictures.py")


    print(f"结束了图片{i}的循环并保存图片")

    with open("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/progress_log.txt", "a") as log_file:
        log_file.write(f"结束了图片{i}的循环并保存图片\n")

    print("=" * 50)

    # 在循环结束时，更新上一次的图片编号
    previous_i = i
