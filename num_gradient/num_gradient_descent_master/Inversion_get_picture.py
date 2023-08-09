import torch
import torch.nn as nn
import torch.optim as optim
from t_ngd_cifar10 import *
from t_ngd_cifar10 import test_classifier
import argparse

parser = argparse.ArgumentParser(description='Inversion_get_picture')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
args = parser.parse_args()
def model_inversion(target_vector, net, learning_rate, num_iterations):
    # 定义生成图像为可训练参数
    generated_image = nn.Parameter(torch.randn_like(target_vector, requires_grad=True))

    # 定义优化器
    optimizer = optim.SGD([generated_image], lr=learning_rate)

    # 循环迭代优化
    for _ in range(num_iterations):
        # 将生成图像输入模型并获得预测分数
        scores = net(generated_image.unsqueeze(0))

        # 计算生成图像与目标向量之间的损失函数
        loss = torch.mean(torch.square(scores - target_vector))

        # 使用反向传播计算梯度并更新生成图像
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 返回生成的图像
    return generated_image.detach()

# 使用示例
target_vector = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 目标向量，例如 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
generated_image = model_inversion(target_vector, net, learning_rate=0.01, num_iterations=1000)
