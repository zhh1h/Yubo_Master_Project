from __future__ import print_function
from ngd_cifar10 import net
from ngd_cifar10 import classes
from ngd_cifar10 import device
from ngd_cifar10 import transform_fn
from ngd_cifar10 import testloader
import sys

#from knockoff.models.cifar import VGG

import knockoff.config as cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision.transforms as transforms
import numpy as np

import os
import argparse


from PIL import Image

import ngd_attacks as ngd



#width, height = (32, 32)
width, height = (200, 200)
data_dir = '/home/yubo/yubo_tem_code/knockoffnets/num_gradient/data/caltech256/256_ObjectCategories'
parser = argparse.ArgumentParser(description='Caltech256 Security Attacks')
parser.add_argument('--input-pic', '-i', type=str, help='Input image', required=False)
parser.add_argument('--target', type=str, help='Target class', required=False)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',default=cfg.MODEL_DIR)
parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
args = parser.parse_args()
params = vars(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

	# Load checkpoint.
print('==> Resuming from checkpoint..')
# checkpoint = torch.load('./checkpoint_lenet/ckpt.t8', map_location=torch.device('cpu'))
# checkpoint = torch.load('./checkpoint_resnet/ckpt.t8', map_location=torch.device('cpu'))
# checkpoint = torch.load('./checkpoint/ckpt_vgg19.t9')#, map_location=torch.device('cpu'))
# checkpoint = torch.load('./checkpoint/ckpt_vggblack.t9', map_location=torch.device('cpu'))

# print(checkpoint)
# net.load_state_dict(checkpoint['net'])
# est_acc = checkpoint['acc']
# print('Resumed')
# tart_epoch = checkpoint['epoch']


# #def test(f=net):
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = f(inputs)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             # print("{} -- {}".format(targets, predicted))
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             print('Test: Loss: %.3f | Accuracy: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def save_img(img, count=None):
    if count != None:
        img = transforms.ToPILImage()(img)
        img.save('output{}.jpg'.format(count))


def test_classifier(h, w, x):
    # x *= 255
    # (x)
    pixels = x.reshape((h, w, 3)).astype('uint8')

    img = Image.fromarray(pixels, mode='RGB')
    img = transform_fn(img)
    img_cuda = img.cuda()
    print(img)

    output = net(img_cuda.unsqueeze(dim=0))

    # print(output[0])
    # print(len(output))
    # print(len(output[0]))
    output = F.softmax(output[0], dim=0)

    print(output)
    save_img(img, count=0)

    value, index = torch.max(output, 0)
    # cifar10
    # print("{} -- {}".format(value, classes[index]))
    # example_class_name = caltech256.classes[index]
    # print(len(classes))
    # print(index)
    print("{} -- {}".format(value, classes[index]))


def save_transform(h, w, x, save_img=None):
    # x *= 255
    img = x.reshape((h, w, 3)).astype('uint8')
    img = Image.fromarray(img, mode='RGB')
    img.save('output.jpg')
    if save_img != None:
        img.save('imgs/output{}.jpg'.format(save_img))
    img = Image.open('output.jpg')
    img = transform_fn(img)
    return img


def create_f(h, w, target):
    def f(x, save_img=None):
        pixels = save_transform(h, w, x, save_img)
        pixels_cuda = pixels.cuda()
        output = net(pixels_cuda.unsqueeze(dim=0))
        output = F.softmax(output[0], dim=0)
        return output[target].item()

    # return lambda x: f(x, target)
    return f


def linearize_pixels(img):
    x = np.copy(np.asarray(img))
    h, w, c = x.shape
    img_array = x.reshape(h * w * c).astype('float64')
    # img_array /= 255
    return h, w, img_array


if args.input_pic:
    net.eval()
    print("There is input pic")
    img = Image.open(args.input_pic)
    h, w, img_array = linearize_pixels(img)

    # ith torch.autograd.profiler.profile(use_cuda=True) as prof:
    test_classifier(h, w, img_array)
    # print(prof)

    if args.target:
        f = create_f(h, w, classes.index(args.target))

        print(f(img_array))

        ngd.num_ascent(f, img_array)



else:
    print("No input pic provided.")
    # You can call test(f=function_to_be_called_for_predictions) to test the accuracy, the default is f=net.







