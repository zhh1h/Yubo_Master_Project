import os
import cv2


def resize_all_images(folder_a, folder_b):
    # 获取文件夹a中的所有jpg图片
    image_files = [f for f in os.listdir(folder_a) if f.endswith('.jpg')]

    # 确保输出文件夹存在
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)

    # 对文件夹a中的每张图片执行操作
    for idx, image_file in enumerate(image_files, 1):
        # 读取图片
        img_path = os.path.join(folder_a, image_file)
        img = cv2.imread(img_path)

        # 使用双线性插值缩放图片到32x32像素
        resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)

        # 保存图片到文件夹b，文件名为文件夹a的名称 + 序号，格式为png
        output_filename = os.path.basename(folder_a) + '_' + str(idx) + '.png'
        output_path = os.path.join(folder_b, output_filename)
        cv2.imwrite(output_path, resized_img)


# 使用方法
folder_a = '/home/yubo/yubo_tem_code/knockoffnets/data/256_ObjectCategories/065.elk'
folder_b = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/caltech256AimImage/065'
resize_all_images(folder_a, folder_b)

