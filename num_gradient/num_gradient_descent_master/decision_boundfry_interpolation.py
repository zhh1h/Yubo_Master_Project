import numpy as np
from PIL import Image
from t_ngd_cifar10 import test_classifier, linearize_pixels
import matplotlib.pyplot as plt

# 选择两个类别的代表性样本
deer_sample = Image.open('./data/cifar_pictures/NO.15class4deer.jpg')  # 这里需要替换为实际的路径
ship_sample = Image.open('./data/cifar_pictures/NO.2class8ship.jpg')  # 同样需要替换为实际的路径


# 显示并分类代表性样本
def display_and_classify(sample, label):
    h, w, img_array = linearize_pixels(sample)
    identified_class = test_classifier(h, w, img_array)
    plt.imshow(sample)
    plt.title(f"{label} Representative Sample, Classified as: {identified_class}")
    plt.show()

display_and_classify(deer_sample, "deer")
display_and_classify(ship_sample, "ship")

# 将这些样本转化为numpy数组
deer_array = np.array(deer_sample)
ship_array = np.array(ship_sample)

# 在这两个样本之间进行线性插值
steps = 10  # 您可以更改此值以获取更多或更少的插值步骤
alpha_values = np.linspace(0, 1, steps)
interpolated_samples = [(1 - alpha) * deer_array + alpha * ship_array for alpha in alpha_values]

# 将插值样本输入到模型中并观察输出
for idx, sample in enumerate(interpolated_samples):
    h, w, img_array = linearize_pixels(Image.fromarray(np.uint8(sample)))
    identified_class = test_classifier(h, w, img_array)

    # 显示插值样本和模型的输出
    plt.imshow(np.uint8(sample))
    plt.title(f"Alpha: {alpha_values[idx]}, Classified as: {identified_class}")
    plt.show()
