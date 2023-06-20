import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse


from models import *
import torchvision

parser = argparse.ArgumentParser(description='CIFAR10 Security Attacks')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
parser.add_argument('--target', type=str, help='Target class', required=False)
args = parser.parse_args()






USE_PGD = True


def draw(data):
    ex = data.squeeze().detach().cpu().numpy()
    plt.imshow(ex, cmap="gray")
    plt.show()


def test(model, device, test_loader, epsilon, t=5, debug=False):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True  # 以便对输入求导 ** 重要 **
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():  # 如果不扰动也预测不对，则跳过
            continue
        if debug:
            draw(data)

        if USE_PGD:
            alpha = epsilon / t  # 每次只改变一小步
            perturbed_data = data
            final_pred = init_pred
            # while target.item() == final_pred.item(): # 只要修改成功就退出
            for i in range(t):  # 共迭代 t 次
                if debug:
                    print("target", target.item(), "pred", final_pred.item())
                loss = F.cross_entropy(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                data_grad = data.grad.data  # 输入数据的梯度 ** 重要 **

                sign_data_grad = data_grad.sign()  # 取符号（正负）
                perturbed_image = perturbed_data + alpha * sign_data_grad  # 添加扰动
                perturbed_data = torch.clamp(perturbed_image, 0, 1)  # 把各元素压缩到[0,1]之间

                output = model(perturbed_data)  # 代入扰动后的数据
                final_pred = output.max(1, keepdim=True)[1]  # 预测选项
                if debug:
                    draw(perturbed_data)
        else:
            loss = F.cross_entropy(output, target)
            model.zero_grad()
            loss.backward()

            data_grad = data.grad.data  # 输入数据的梯度 ** 重要 **
            sign_data_grad = data_grad.sign()  # 取符号（正负）
            perturbed_image = data + epsilon * sign_data_grad  # 添加扰动
            perturbed_data = torch.clamp(perturbed_image, 0, 1)  # 把各元素压缩到[0,1]之间

            output = model(perturbed_data)  # 代入扰动后的数据
            final_pred = output.max(1, keepdim=True)[1]

        # 统计准确率并记录，以便后面做图
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:  # 保存扰动后错误分类的图片
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))  # 计算整体准确率
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples


epsilons = [0, .05, .1, .15, .2, .25, .3]  # 使用不同的调整力度
device='cuda'
net = VGG('VGG19')
net = net.to(device)

if device == 'cuda':
   torch.cuda.set_device('cuda:0')
net = torch.nn.DataParallel(net).cuda()

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


#test_loader = testloader
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
test_loader = testloader

print('==> Resuming from checkpoint..')
pretrained_model = torch.load('./pytorch_cifar_master/checkpoint/ckpt.pth', map_location=torch.device('cuda:0'))  # 使用的预训练模型路径
print(pretrained_model)
net.load_state_dict(pretrained_model['net'])
est_acc = pretrained_model['acc']
print('Resumed')
criterion = nn.CrossEntropyLoss()


model = net
model.eval()

accuracies = []
examples = []






for eps in epsilons:  # 每次测一种超参数
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


plt.figure(figsize=(8,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        ex = ex.transpose((1, 2, 0))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()