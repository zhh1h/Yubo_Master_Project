'''CIFAR10 Security POC
Tiago Alves <tiago@ime.uerj.br>'''
from __future__ import print_function
import time
#
#
#from knockoff.models.cifar import vgg19
from knockoff.models.cifar import vgg19
import sys
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/knockoff")
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/")
# #from num_gradient.num_gradient_descent_master

import torch
import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#
import torchvision
from torchvision.models import googlenet
import torchvision.transforms as transforms
import numpy as np
from pytorch_cifar_master.models import VGG
#
import os
import argparse
#
from models import *
from utils import progress_bar
#
from PIL import Image
#
import ngd_attacks as ngd
#import pgd
#
width, height = (32, 32)
import torch.optim as optim
#
#
#
parser = argparse.ArgumentParser(description='CIFAR10 Security Attacks')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
parser.add_argument('--target', type=str, help='Target class', required=False)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
#
#
#
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#device='cuda:0'
device='cuda'

#GoogLeNet = googlenet(pretr
#device = 'cpu'
# Model
#print('==> Building model..')
net = vgg19(num_classes=10)
#net = ResNet18()
net = net.to(device)
# fc_in_features = net.fc.in_features
# net.fc = torch.nn.Linear(fc_in_features,10)

# net = PreActResNet18()
#net = GoogLeNet()
#net = googlenet(pretrained=True)
#net = DenseNet121()
# net = ResNeXt29_2x64d()
#net = MobileNet()
#net = QuantMobileNetV2()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
#net = net.to(device)
if device == 'cuda':
   torch.cuda.set_device('cuda:0')
#net = torch.nn.DataParallel(net).cuda()
#cudnn.benchmark = True

    # Load checkpoint.
print('==> Resuming from checkpoint..')
#checkpoint = torch.load('./checkpoint_lenet/ckpt.t8', map_location=torch.device('cuda:0'))
####checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=torch.device('cuda:0'))
checkpoint = torch.load('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/victim/cifar10-vgg19/checkpoint.pth.tar', map_location='cuda:0')

#checkpoint = torch.load('./checkpoint/ckpt_resnet18_rgb.t9', map_location=torch.device('cuda:0'))
#checkpoint = torch.load('./checkpoint/ckpt_googlenet_rgb.t9', map_location=torch.device('cuda:0'))

#checkpoint = torch.load('./checkpoint/ckpt_quantizablemobilenetv2.cpt', map_location=torch.device('cuda'))
#checkpoint = torch.load('./checkpoint/ckpt_mobilenet_quant.t7', map_location=torch.device('cpu'))
#checkpoint = torch.load('./checkpoint/ckpt_vgg19.t9', map_location=torch.device('cuda'))
#checkpoint = torch.load('./checkpoint/ckpt_googlenet.cpt', map_location=torch.device('cuda'))

#print(checkpoint)
#net.load_state_dict(checkpoint['net'])
net.load_state_dict(checkpoint['state_dict'])
#print(checkpoint.keys())
best_acc = checkpoint['best_acc']
print(best_acc)
print('Resumed')
#tart_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

net.eval()
# def preprocess_image(h, w, x):
#     x = x.astype('uint8')
#     pixels = x.reshape((h, w, 3))
#     img = Image.fromarray(pixels, mode='RGB')
#     # ... any other preprocessing steps that you had in test_classifier should go here ...
#     # For example:
#     #img = save_transform(save_img=None)
#     img_tensor = torch.Tensor(np.array(img)).permute(2, 0, 1) / 255.0
#     return img_tensor

# def preprocess_with_transform_fn(h, w, x):
#     x = x.astype('uint8').reshape((h, w, 3))
#     img = Image.fromarray(x, mode='RGB')
#     img_tensor = transform_fn(img)
#     return img_tensor


def test(f=net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print("{} -- {}".format(targets, predicted))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#
def save_img(img, count=None):
    if not isinstance(img, (torch.Tensor, np.ndarray)):
        # 如果 img 已经是一个 PIL 图像，则直接保存
        img.save(f'output{count}.png' if count is not None else 'output_DB.png')
    else:
        # 否则，将其转换为 PIL 图像并保存
        img = transforms.ToPILImage()(img)
        img.save(f'output{count}.png' if count is not None else 'output_DB.png ')





#@profile
def test_classifier(h, w, x, preprocessed = False, return_class_index=False, return_confidence=False):
    # if not preprocessed:
    #     img_tensor = preprocess_with_transform_fn(h, w, x)  # 使用新的预处理函数
    #     print(type(img_tensor))
    #
    # else:
    #     img_tensor = x

    #img_tensor = preprocess_image(h, w, x)

    pixels = x.reshape((h, w, 3)).astype('uint8')
    img = Image.fromarray(pixels, mode='RGB')
    img = transform_fn(img)
    save_img(img, count=0)
    img_cuda = img.cuda()
    # 	print(img)
    #
    # 	output = net(img_cuda.unsqueeze(dim=0))

    # 使用预处理后的张量进行分类
    net.eval()
    # print(type(img_tensor))

    output = net(img_cuda.unsqueeze(dim=0))
    output_softmax = F.softmax(output[0], dim=0)
    # save_img(img, count=0)

    # 其他部分保持不变
    value, index = torch.max(output_softmax, 0)
    predicted_class = classes[index]
    print("{} -- {}".format(value, predicted_class))
    #print(f"output_1:{output}")
    print(f"output:{output_softmax}")
    #save_img(img, count=0)
    if return_class_index and return_confidence:
        return predicted_class, index, value.item()
    elif return_class_index:
        return predicted_class, index
    elif return_confidence:
        return predicted_class, value.item()
    else:
        return predicted_class
    #print("{} -- {}".format(value, classes[index]))
    # print(f"output_1:{output}")
    # print(f"output:{output_softmax}")


def save_transform(h, w, x, save_img=None):
    # if isinstance(x, torch.Tensor):
    #     x = x.to(torch.uint8).cpu().numpy()
    #
    #     # 将图像数组重塑为 (h, w, 3)
    # img_data = x.reshape((h, w, 3)).astype('uint8')
    #
    # # 使用PIL创建图像
    # img_to_save = Image.fromarray(img_data, 'RGB')
    #
    # # 保存图像
    # img_to_save.save('output.jpg')
    #
    # # 如果提供了save_img参数，以另一个名称保存图像
    # if save_img is not None:
    #     img_to_save.save(f'imgs/output{save_img}.jpg')
    #
    # return img_to_save  # 如果需要，也可以返回保存的图像
    #x *= 255
    #img = x.reshape((h, w, 3)).astype('uint8')
    if isinstance(x, torch.Tensor):
        img = x.to(torch.uint8).cpu().numpy().reshape((h, w, 3))
    else:  # x is a numpy ndarray
        img = x.reshape((h, w, 3)).astype(np.uint8)


    img = Image.fromarray(img, mode='RGB')
    #img = transform_fn(img)
    img.save('output_DB.png')
    img = transform_fn(img)
    if save_img != None:
        img.save('imgs/output{}.png'.format(save_img))
    #img = Image.open('output.jpg')
    #img = transform_fn(img)
    return img

def create_f(h, w, target_class_name):
    def f(x, save_img=None, check_prediction=False):
        # Preprocess the image
        pixels = save_transform(h, w, x, save_img)
        pixels_cuda = pixels.cuda()
        net.eval()
        output = net(pixels_cuda.unsqueeze(dim=0))
        output_softmax = F.softmax(output[0], dim=0)
        conf, predicted_index = torch.max(output_softmax, 0)

        target_class_index = classes.index(target_class_name)
        return output_softmax[target_class_index].item(), predicted_index.item()

    return f



# def create_f(h, w, target):
#  	def f(x, save_img=None):
#  		pixels = save_transform(h, w, x, save_img)
#  		pixels_cuda = pixels.cuda()
#  		output = net(pixels_cuda.unsqueeze(dim=0))
#  		output = F.softmax(output[0], dim=0)
#         if check_prediction:
#             conf_predicted, predicted = torch.max(output, 0)
#              print("target: {} predicted: {}".format(classes[target], classes[predicted]))
#              if predicted != target:
#                  return 0
#          return output[target].item()
#      return f


# def linearize_pixels(img):
#     x = np.copy(np.asarray(img))
#     h, w, c = x.shape
#     img_array = x.reshape(h*w*c).astype('float64')
#     #img_array /= 255
#     return h, w, img_array

def linearize_pixels(img):
    x = np.copy(np.asarray(img))
    h, w, c = x.shape
    img_array = x.reshape(h * w * c).astype('float64')
    return h, w, img_array



# def linearize_pixels(img):
#     # 将图像转换为 NumPy 数组
#     x = np.array(img)
#
#     # 获取图像的维度
#     h, w, c = x.shape
#
#     # 确保数据类型和范围与其他预处理步骤相同
#     # 在这里，我假设其他预处理步骤需要 [0, 1] 范围的 float64 类型。
#     # 如果其他预处理步骤使用的是 [0, 255] 范围的 uint8 类型，那么这里就不需要除以 255。
#     #x = x.astype('float64') / 255.0
#
#     # 按照（channel, height, width）的顺序重塑数组
#     x = np.transpose(x, (2, 0, 1))
#
#     # 展平数组
#     img_array = x.ravel()
#
#     return h, w, img_array

def num_grad(f, x):
    delta = 20
    grad = np.zeros_like(x)  # Match the shape of x
    perturbed_x = np.copy(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        original_value = x[ix]

        # Perturb the current element of x
        perturbed_x[ix] = original_value + delta
        grad[ix] = (f(perturbed_x)[0] - f(x)[0]) / delta  # 只处理置信度

        # Reset the perturbed value
        perturbed_x[ix] = original_value

        it.iternext()

    return grad


def pppgd(f, x, target, num_steps=100, initial_alpha=0.5, momentum=0.9):
    target_index = classes.index(target)
    conf, predicted_class = f(x)
    print(f"Initial confidence for target '{target}': {conf}, predicted class: {classes[predicted_class]}")

    alpha = initial_alpha
    grad = num_grad(f, x)
    sign_data_grad = torch.sign(torch.from_numpy(grad))
    update = torch.zeros_like(sign_data_grad)

    for i in range(num_steps):
        x = torch.from_numpy(x)
        update = momentum * update + alpha * sign_data_grad
        x = x + update
        x = x.detach().numpy()
        conf, predicted_class = f(x)

        print(f"Step {i+1}, confidence {conf}, predicted class: {classes[predicted_class]}")

        if predicted_class == target_index and conf > 0.2:
            print(f"Target '{target}' reached confidence {conf} at step {i+1}, predicted class: {classes[predicted_class]}")
            break
        alpha *= 0.99

    if predicted_class != target_index or conf <= 0.2:
        print(f"Failed to reach confidence > 0.2 for image on target class {target}")
        return pppgd(f, x, target, num_steps, initial_alpha, momentum)  # 重新执行攻击直到条件满足

    print(f"Final confidence for target '{target}': {conf}, predicted class: {classes[predicted_class]}")
    return x, conf






# ... [前面的代码保持不变] ...

# 图片文件夹路径
image_folder = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/fix_pixels_images"
output_folder = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/DBImages"  # 输出文件夹路径

# ... [之前的代码保持不变] ...

# 日志文件路径
# ... [之前的代码保持不变] ...

# 日志文件路径
log_file_path = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/DB_log.txt"

# 自动遍历图片文件夹并进行处理
with open(log_file_path, "a") as log_file:
    for i in range(247, 257):  # 假设有256张图片
        image_path = os.path.join(image_folder, f"fix_pixels{i}.jpg")

        if not os.path.exists(image_path):
            log_msg = f"Image not found: {image_path}\n"
            print(log_msg)
            log_file.write(log_msg)
            continue

        try:
            img = Image.open(image_path)
            h, w, img_array = linearize_pixels(img)

            # 遍历每个目标类别
            for target_class in classes:
                print(f"Starting pppgd attack on image {i} for target class '{target_class}'")
                f = create_f(h, w, target_class)
                img_array, conf = pppgd(f, img_array, target_class, num_steps=10)
                # ... [处理和保存图像的代码] ...

                if conf > 0.2:
                    # 保存图片
                    reshaped_img_array = img_array.reshape((h, w, 3)).astype('uint8')
                    output_filename = f"{target_class}_{conf:.4f}.png"
                    final_img = Image.fromarray(reshaped_img_array, 'RGB')
                    final_img.save(os.path.join(output_folder, output_filename))

                    log_msg = f"Processed image {image_path} for target class {target_class}, saved as {output_filename}\n"
                    print(log_msg)
                    log_file.write(log_msg)
                else:
                    log_msg = f"Failed to reach confidence > 0.2 for image {image_path} on target class {target_class}\n"
                    print(log_msg)
                    log_file.write(log_msg)

            log_msg = f"Completed processing of image: {image_path}\n"
            print(log_msg)
            log_file.write(log_msg)

        except FileNotFoundError:
            log_msg = f"Failed to open image: {image_path}\n"
            print(log_msg)
            log_file.write(log_msg)
            continue

# ... [之后的

# ... [之后的代码保持不变] ...


# ... [剩余的代码保持不变] ...


# if args.input_pic:
#         #print("There is input pic")
#         #net.eval()
#         img = Image.open(args.input_pic)
#         h, w, img_array = linearize_pixels(img)
#
#         # with torch.autograd.profiler.profile(use_cuda=True) as prof:
#         #     import pyprof
#         with torch.autograd.profiler.emit_nvtx():
#             net.eval()
#         test_classifier(h, w, img_array)
#         #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
#         #print(prof)
#
#         if args.target:
#             f = create_f(h, w, classes.index(args.target))
#
#             start_time0 = time.time()
#             img_array, conf = pppgd(f, img_array, num_steps=10)  # 调用新的pppgd函数
#             predicted_class, _, _ = test_classifier(h, w, img_array, return_class_index=True, return_confidence=True)
#             reshaped_img_array = img_array.reshape((h, w, 3)).astype('uint8')
#
#             # 保存满足条件的图片到指定文件夹
#             output_folder = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/DBImages"  # 替换为您指定的文件夹路径
#             output_filename = f"{predicted_class}_{conf:.2f}.png"  # 文件命名为“种类_分数.png”
#             final_img = Image.fromarray(reshaped_img_array, 'RGB')
#             final_img.save(os.path.join(output_folder, output_filename))
#
#             final_time = time.time()
#             final_interval = final_time - start_time0
#             print(f"final time interval: {final_interval} 秒")
#
#
#
#                 #break
#                 #return img_array
#             # #ngd.ppgd(f,img_array)
#
#
#
#
# else:
#         print("No input pic provided")

