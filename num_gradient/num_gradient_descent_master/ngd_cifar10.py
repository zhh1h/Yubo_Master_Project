from __future__ import print_function
#from knockoff.models.cifar import VGG
#from knockoff.models.cifar import resnet34
#from knockoff.models.imagenet import resnet34
#import knockoff.models.imagenet
import torchvision.models as models
from torch.utils.data import Dataset
import sys

sys.path.append("/home/yubo/yubo_tem_code/knockoffnets")
sys.path.append("/home/yubo/yubo_tem_code/knockoffnets/num_gradient")


from knockoff.datasets.caltech256 import Caltech256
#import knockoff.utils.model as model_utils
import knockoff.config as cfg
#import os.path as osp
#import json
#from datetime import datetime


'''CIFAR10 Security POC
Tiago Alves <tiago@ime.uerj.br>'''

'''This portion of the code is based on the example provided Liu Kuan (https://github.com/kuangliu/pytorch-cifar). 
However, the attacks/defenses in ngd_attacks.py can be used in any other implementations of CIFAR10 classifiers with almost no adaption required.'''
#from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

from PIL import Image

import ngd_attacks as ngd


#width, height = (32, 32)
width, height = (200, 200)
data_dir = '/home/yubo/yubo_tem_code/knockoffnets/num_gradient/data/caltech256/256_ObjectCategories'



#parser = argparse.ArgumentParser(description='CIFAR10 Security Attacks')
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

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# cifar10
# transform_test = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

#caltech256
transform_fn = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



#caltech256
transform_test = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

#cifar10
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#caltech256
#caltech256 = datasets.Caltech256(root='./data', download=False)
#root='/home/yubo/yubo_tem_code/knockoffnets/num_gradient/data'
#testset = torchvision.datasets.Caltech256(root='./data',train = False,download=True,transform = transform_test)
trainset = Caltech256(train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#root='/home/yubo/yubo_tem_code/knockoffnets/num_gradient/data'
testset = Caltech256( train=False, transform=transform_test)

#testset = caltech256(train = False,transform = transform_test)
#testset = datasets.Caltech256(root='./data', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

#cifar10 class
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#caltech256

#classes = testset.classes
classes = os.listdir(data_dir)
#classes = sorted(os.listdir(data_dir))

#cifar10

# transform_fn = transforms.Compose([
# 	transforms.Resize(32),
# 	transforms.CenterCrop(32),
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#])


# Model
#print('==> Building model..')
#net = VGG('VGG19')
net =models.resnet34()
fc_in_features = net.fc.in_features
net.fc = torch.nn.Linear(fc_in_features,256)
#net =resnet34()
#net = model
#net = ResNet18()
# net = PreActResNet18()
#net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
#net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
	net = torch.nn.DataParallel(net)


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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=170)




# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
#	cudnn.benchmark = True

	# Load checkpoint.
print('==> Resuming from checkpoint..')
# checkpoint = torch.load('./checkpoint_lenet/ckpt.t8', map_location=torch.device('cpu'))
#checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=torch.device('cpu'))
# checkpoint = torch.load('./checkpoint/ckpt_vgg19.t9')#, map_location=torch.device('cpu'))
# checkpoint = torch.load('./checkpoint/ckpt_vggblack.t9', map_location=torch.device('cpu'))

print(checkpoint)
net.load_state_dict(checkpoint['net'])
est_acc = checkpoint['acc']
print('Resumed')
tart_epoch = checkpoint['epoch']





# Training

# out_path = params['out_path']
# model_utils.train_model(net, trainset, out_path,testset=testset,device=device)

# params['created_on'] = str(datetime.now())
# params_out_path = osp.join(out_path, 'params.json')
# with open(params_out_path, 'w') as jf:
# 	json.dump(params, jf, indent=True)
#**params
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('Train: Loss: %.3f | Accuracy: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
#
#
def test(epoch):
    global best_acc
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
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Test: Loss: %.3f | Accuracy: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



#
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+2):
    train(epoch)
    test(epoch)
    scheduler.step()




# def test(f=net):
# 	net.eval()
# 	test_loss = 0
# 	correct = 0
# 	total = 0
# 	with torch.no_grad():
# 		for batch_idx, (inputs, targets) in enumerate(testloader):
# 			inputs, targets = inputs.to(device), targets.to(device)
# 			outputs = f(inputs)
# 			loss = criterion(outputs, targets)
#
# 			test_loss += loss.item()
# 			_, predicted = outputs.max(1)
# 			#print("{} -- {}".format(targets, predicted))
# 			total += targets.size(0)
# 			correct += predicted.eq(targets).sum().item()
#
# 			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
# 			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#
#
# def save_img(img, count=None):
# 	if count != None:
# 		img = transforms.ToPILImage()(img)
# 		img.save('output{}.jpg'.format(count))
#
#
#
#
#
#
# def test_classifier(h, w, x):
# 	#x *= 255
# 	#(x)
# 	pixels = x.reshape((h, w, 3)).astype('uint8')
#
# 	img = Image.fromarray(pixels, mode='RGB')
# 	img = transform_fn(img)
# 	img_cuda = img.cuda()
# 	print(img)
#
# 	output = net(img_cuda.unsqueeze(dim=0))
#
# 	#print(output[0])
# 	#print(len(output))
# 	#print(len(output[0]))
# 	output = F.softmax(output[0], dim=0)
#
# 	print(output)
# 	save_img(img, count=0)
#
# 	value, index = torch.max(output, 0)
# 	#cifar10
# 	#print("{} -- {}".format(value, classes[index]))
#     #example_class_name = caltech256.classes[index]
# 	#print(len(classes))
# 	#print(index)
# 	print("{} -- {}".format(value, classes[index]))
#
# def save_transform(h, w, x, save_img=None):
# 	#x *= 255
# 	img = x.reshape((h, w, 3)).astype('uint8')
# 	img = Image.fromarray(img, mode='RGB')
# 	img.save('output.jpg')
# 	if save_img != None:
# 		img.save('imgs/output{}.jpg'.format(save_img))
# 	img = Image.open('output.jpg')
# 	img = transform_fn(img)
# 	return img
#
# def create_f(h, w, target):
# 	def f(x, save_img=None):
# 		pixels = save_transform(h, w, x, save_img)
# 		pixels_cuda = pixels.cuda()
# 		output = net(pixels_cuda.unsqueeze(dim=0))
# 		output = F.softmax(output[0], dim=0)
# 		return output[target].item()
# 	#return lambda x: f(x, target)
# 	return f
#
#
#
#
# def linearize_pixels(img):
# 	x = np.copy(np.asarray(img))
# 	h, w, c = x.shape
# 	img_array = x.reshape(h*w*c).astype('float64')
# 	#img_array /= 255
# 	return h, w, img_array
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# if args.input_pic:
# 	net.eval()
# 	print("There is input pic")
# 	img = Image.open(args.input_pic)
# 	h, w, img_array = linearize_pixels(img)
#
#
# 	#ith torch.autograd.profiler.profile(use_cuda=True) as prof:
# 	test_classifier(h, w, img_array)
# 	#print(prof)
#
# 	if args.target:
# 		f = create_f(h, w, classes.index(args.target))
#
# 		print(f(img_array))
#
# 		ngd.num_ascent(f, img_array)
#
#
#
# else:
# 	print("No input pic provided.")
#         #You can call test(f=function_to_be_called_for_predictions) to test the accuracy, the default is f=net.
#
#
#
#
#
#
#
