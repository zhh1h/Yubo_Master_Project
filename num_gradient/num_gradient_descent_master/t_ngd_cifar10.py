'''CIFAR10 Security POC
Tiago Alves <tiago@ime.uerj.br>'''
from __future__ import print_function
import time
#
#
from knockoff.models.cifar import vgg19
#import sys
#sys.path.append("/home/yubo/yubo_tem_code/knockoffnets/num_gradient/master_1/pytorch_cifar_master/")
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
    transforms.ToTensor()])#,
#	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#])
#device='cuda:0'
device='cuda'

#GoogLeNet = googlenet(pretr
#device = 'cpu'
# Model
#print('==> Building model..')
net = VGG('VGG19')
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
net = torch.nn.DataParallel(net).cuda()
#cudnn.benchmark = True

    # Load checkpoint.
print('==> Resuming from checkpoint..')
#checkpoint = torch.load('./checkpoint_lenet/ckpt.t8', map_location=torch.device('cuda:0'))
checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=torch.device('cuda:0'))
#checkpoint = torch.load('./checkpoint/ckpt_resnet18_rgb.t9', map_location=torch.device('cuda:0'))
#checkpoint = torch.load('./checkpoint/ckpt_googlenet_rgb.t9', map_location=torch.device('cuda:0'))

#checkpoint = torch.load('./checkpoint/ckpt_quantizablemobilenetv2.cpt', map_location=torch.device('cuda'))
#checkpoint = torch.load('./checkpoint/ckpt_mobilenet_quant.t7', map_location=torch.device('cpu'))
#checkpoint = torch.load('./checkpoint/ckpt_vgg19.t9', map_location=torch.device('cuda'))
#checkpoint = torch.load('./checkpoint/ckpt_googlenet.cpt', map_location=torch.device('cuda'))

print(checkpoint)
net.load_state_dict(checkpoint['net'])
est_acc = checkpoint['acc']
print('Resumed')
#tart_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

net.eval()
def preprocess_image(h, w, x):
    x = x.astype('uint8')
    pixels = x.reshape((h, w, 3))
    img = Image.fromarray(pixels, mode='RGB')
    # ... any other preprocessing steps that you had in test_classifier should go here ...
    # For example:
    #img = save_transform(save_img=None)
    img_tensor = torch.Tensor(np.array(img)).permute(2, 0, 1) / 255.0
    return img_tensor


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
        img.save(f'output{count}.jpg' if count is not None else 'output.jpg')
    else:
        # 否则，将其转换为 PIL 图像并保存
        img = transforms.ToPILImage()(img)
        img.save(f'output{count}.jpg' if count is not None else 'output.jpg')





#@profile
def test_classifier(h, w, x, preprocessed=False, return_class_index=False, return_confidence=False):
    if not preprocessed:
        img_tensor = preprocess_image(h, w, x)
    else:
        img_tensor = x
    #x *= 255


    #img_tensor = preprocess_image(h, w, x)
    #save_img(img_tensor, count=0)

    # 使用预处理后的张量进行分类
    net.eval()
    output = net(img_tensor.unsqueeze(dim=0))
    output_softmax = F.softmax(output[0], dim=0)

    # 其他部分保持不变
    value, index = torch.max(output_softmax, 0)
    predicted_class = classes[index]
    print("{} -- {}".format(value, predicted_class))
    #print(f"output_1:{output}")
    print(f"output:{output_softmax}")
    save_img(img, count=0)
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
    if isinstance(x, torch.Tensor):
        x = x.to(torch.uint8).cpu().numpy()

        # 将图像数组重塑为 (h, w, 3)
    img_data = x.reshape((h, w, 3)).astype('uint8')

    # 使用PIL创建图像
    img_to_save = Image.fromarray(img_data, 'RGB')

    # 保存图像
    img_to_save.save('output.jpg')

    # 如果提供了save_img参数，以另一个名称保存图像
    if save_img is not None:
        img_to_save.save(f'imgs/output{save_img}.jpg')

    return img_to_save  # 如果需要，也可以返回保存的图像
    # #x *= 255
    # #img = x.reshape((h, w, 3)).astype('uint8')
    # if isinstance(x, torch.Tensor):
    #     img = x.to(torch.uint8).cpu().numpy().reshape((h, w, 3))
    # else:  # x is a numpy ndarray
    #     img = x.reshape((h, w, 3)).astype(np.uint8)
    #
    #
    # img = Image.fromarray(img, mode='RGB')
    # img.save('output.jpg')
    # if save_img != None:
    #     img.save('imgs/output{}.jpg'.format(save_img))
    # img = Image.open('output.jpg')
    # img = transform_fn(img)
    # return img

def create_f(h, w, target):
    def f(x, save_img=None, check_prediction=False):
        # Preprocess the image
        #pixels = save_transform(h, w, x, save_img)
        save_transform(h, w, x, save_img)
        img_tensor = preprocess_image(h, w, x)
        #img_tensor = save_transform(h, w, x, save_img)
        output = net(img_tensor.unsqueeze(dim=0))
        output = F.softmax(output[0], dim=0)
        if check_prediction:
            conf_predicted, predicted = torch.max(output, 0)
            print("target: {} predicted: {}".format(classes[target], classes[predicted]))
            if predicted != target:
                return 0
        return output[target].item()
    return f





def linearize_pixels(img):
    x = np.copy(np.asarray(img))
    h, w, c = x.shape
    img_array = x.reshape(h*w*c).astype('float64')
    #img_array /= 255
    return h, w, img_array

















if args.input_pic:
        #print("There is input pic")
        net.eval()
        img = Image.open(args.input_pic)
        h, w, img_array = linearize_pixels(img)

        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #import pyprof
        #with torch.autograd.profiler.emit_nvtx():
            #net.eval()
        test_classifier(h, w, img_array)
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        #print(prof)

        if args.target:
            f = create_f(h, w, classes.index(args.target))

            #print(f(img_array))
            #img_array = ngd.num_ascent_g(f, img_array)
            start_time0 = time.time()
            #while True:
            start_time = time.time()
                #img_array = ngd.num_ascent(f, img_array)
                #img_array = ngd.num_ascent_g(f, img_array)
                #img_array = ngd.ppgd(f, img_array)
            while f(img_array) <= 0.8:
                img_array = ngd.pppgd(f, img_array,num_steps = 10)
                #img_array = ngd.pgd_d(f, img_array, epsilons=0.3, alpha=0.01, num_steps=5)
                index = test_classifier(h,w,img_array)
                print(index)
                step_time = time.time()
                time_interval = step_time - start_time
                print(f"gradient time: {time_interval} 秒")
            #if ((test_classifier(h, w, img_array)) == classes.index(args.target)) and (create_f(h, w, img_array) >= 0.5):
            #if create_f(h, w, classes.index(args.target)) >= 0.5:
            final_time= time.time()
            final_interval = final_time - start_time0
            print(f"final time interval: {final_interval} 秒")


                #break
                #return img_array
            # #ngd.ppgd(f,img_array)




else:
        print("No input pic provided")


#
#
# if args.input_pic:
#     img = Image.open(args.input_pic)
#     h, w, img_array = linearize_pixels(img)
#
#     # Test the classifier with the original image
#     original_label = test_classifier(h, w, img_array)
#
#     if args.target:
#         # Convert the target label name to its corresponding index
#         target_label_index = classes.index(args.target)
#         f = create_f(h, w, target_label_index)
#
#
#         start_time_total = time.time()  # Start timing for total time
#
#         # Perform PGD attack
#         while True:
#             start_time_pgd = time.time()  # Start timing for PGD
#             #img_array = img_array.reshape(1, 3, h, w)
#             #img_array = img_array.squeeze(0)  # remove the first dimension
#             img_array = ngd.pgd_d(f, img_array, epsilons=0.3, alpha=0.01, num_steps=100)
#             perturbed_label = test_classifier(h, w, img_array)
#             end_time_pgd = time.time()  # End timing for PGD
#             print(f"Time elapsed for PGD: {end_time_pgd - start_time_pgd} seconds")
#             #if perturbed_label == target_label_index:
#             if perturbed_label == target_label_index:
#
#                 break
#
#         end_time_total = time.time()  # End timing for total time
#         print(f"Total time elapsed: {end_time_total - start_time_total} seconds")
#
#         print(f"Original label: {classes[original_label]}, Perturbed label: {classes[perturbed_label]}")
# else:
#     print("No input pic provided")










