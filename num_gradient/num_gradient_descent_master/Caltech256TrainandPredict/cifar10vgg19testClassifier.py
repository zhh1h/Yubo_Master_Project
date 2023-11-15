from __future__ import print_function
import time
import sys
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/master_1/")
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/")
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
#from pytorch_cifar_master.models import VGG
from knockoff.models.cifar import vgg19


from models import *
import ngd_attacks as ngd
torch.nn.Module.dump_patches = True
device = 'cuda'

# Load model
print('==> Building model..')
net = vgg19(num_classes=10)
net = net.to(device)
def print_model_structure(net):
    print(net)

def print_layer_parameters(net):
    for name, param in net.named_parameters():
        print(name, param.size())

# Load checkpoint
print('==> Resuming from checkpoint..')
checkpoint = torch.load('/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/victim/cifar10-vgg19/checkpoint.pth.tar', map_location='cuda:0')


net.load_state_dict(checkpoint['state_dict'])

# 打印模型结构


# Set model to evaluation mode
net.eval()

# Define transformations
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def linearize_pixels(img):
    x = np.copy(np.asarray(img))
    h, w, c = x.shape
    img_array = x.reshape(h * w * c).astype('float64')
    return h, w, img_array

def test_classifier(h, w, x, return_confidence=False, return_all_scores=False):
    pixels = x.reshape((h, w, 3)).astype('uint8')
    img = Image.fromarray(pixels, mode='RGB')
    img = transform_fn(img)
    img_cuda = img.cuda()

    with torch.no_grad():
        output = net(img_cuda.unsqueeze(dim=0))
        output_softmax = torch.nn.functional.softmax(output[0], dim=0)

    value, index = torch.max(output_softmax, 0)
    predicted_class = classes[index]

    if return_confidence and return_all_scores:
        return predicted_class, value.item(), output_softmax.tolist()
    elif return_confidence:
        return predicted_class, value.item()
    elif return_all_scores:
        return predicted_class, output_softmax.tolist()
    else:
        return predicted_class





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
    img.save('output.png')
    img = transform_fn(img)
    if save_img != None:
        img.save('imgs/output{}.png'.format(save_img))
    #img = Image.open('output.jpg')
    #img = transform_fn(img)
    return img
def create_f(h, w, target):
    def f(x, save_img=None, check_prediction=False):
        pixels = save_transform(h, w, x, save_img)
        pixels_cuda = pixels.cuda()

        with torch.no_grad():
            output = net(pixels_cuda.unsqueeze(dim=0))
            output = torch.nn.functional.softmax(output[0], dim=0)

        if check_prediction:
            conf_predicted, predicted = torch.max(output, 0)
            print("target: {} predicted: {}".format(classes[target], classes[predicted]))
            if predicted != target:
                return 0
        return output[target].item()
    return f

