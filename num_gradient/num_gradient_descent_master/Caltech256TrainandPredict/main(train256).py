import argparse
import json
import os
import os.path as osp
import pickle

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from knockoff.models.cifar.vgg import vgg19
import torch.backends.cudnn as cudnn
from models import *

import os
import sys

sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/knockoff")
sys.path.append("/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/master_1/pytorch-cifar-master/models")

parser = argparse.ArgumentParser(description='PyTorch Caltech256 Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()



datasetpath="/home/yubo/yubo_tem_code/knockoffnets/data/256_ObjectCategories"
classes = ('ak47',
 'american-flag',
 'backpack',
 'baseball-bat',
 'baseball-glove',
 'basketball-hoop',
 'bat',
 'bathtub',
 'bear',
 'beer-mug',
 'billiards',
 'binoculars',
 'birdbath',
 'blimp',
 'bonsai-101',
 'boom-box',
 'bowling-ball',
 'bowling-pin',
 'boxing-glove',
 'brain-101',
 'breadmaker',
 'buddha-101',
 'bulldozer',
 'butterfly',
 'cactus',
 'cake',
 'calculator',
 'camel',
 'cannon',
 'canoe',
 'car-tire',
 'cartman',
 'cd',
 'centipede',
 'cereal-box',
 'chandelier-101',
 'chess-board',
 'chimp',
 'chopsticks',
 'cockroach',
 'coffee-mug',
 'coffin',
 'coin',
 'comet',
 'computer-keyboard',
 'computer-monitor',
 'computer-mouse',
 'conch',
 'cormorant',
 'covered-wagon',
 'cowboy-hat',
 'crab-101',
 'desk-globe',
 'diamond-ring',
 'dice',
 'dog',
 'dolphin-101',
 'doorknob',
 'drinking-straw',
 'duck',
 'dumb-bell',
 'eiffel-tower',
 'electric-guitar-101',
 'elephant-101',
 'elk',
 'ewer-101',
 'eyeglasses',
 'fern',
 'fighter-jet',
 'fire-extinguisher',
 'fire-hydrant',
 'fire-truck',
 'fireworks',
 'flashlight',
 'floppy-disk',
 'football-helmet',
 'french-horn',
 'fried-egg',
 'frisbee',
 'frog',
 'frying-pan',
 'galaxy',
 'gas-pump',
 'giraffe',
 'goat',
 'golden-gate-bridge',
 'goldfish',
 'golf-ball',
 'goose',
 'gorilla',
 'grand-piano-101',
 'grapes',
 'grasshopper',
 'guitar-pick',
 'hamburger',
 'hammock',
 'harmonica',
 'harp',
 'harpsichord',
 'hawksbill-101',
 'head-phones',
 'helicopter-101',
 'hibiscus',
 'homer-simpson',
 'horse',
 'horseshoe-crab',
 'hot-air-balloon',
 'hot-dog',
 'hot-tub',
 'hourglass',
 'house-fly',
 'human-skeleton',
 'hummingbird',
 'ibis-101',
 'ice-cream-cone',
 'iguana',
 'ipod',
 'iris',
 'jesus-christ',
 'joy-stick',
 'kangaroo-101',
 'kayak',
 'ketch-101',
 'killer-whale',
 'knife',
 'ladder',
 'laptop-101',
 'lathe',
 'leopards-101',
 'license-plate',
 'lightbulb',
 'light-house',
 'lightning',
 'llama-101',
 'mailbox',
 'mandolin',
 'mars',
 'mattress',
 'megaphone',
 'menorah-101',
 'microscope',
 'microwave',
 'minaret',
 'minotaur',
 'motorbikes-101',
 'mountain-bike',
 'mushroom',
 'mussels',
 'necktie',
 'octopus',
 'ostrich',
 'owl',
 'palm-pilot',
 'palm-tree',
 'paperclip',
 'paper-shredder',
 'pci-card',
 'penguin',
 'people',
 'pez-dispenser',
 'photocopier',
 'picnic-table',
 'playing-card',
 'porcupine',
 'pram',
 'praying-mantis',
 'pyramid',
 'raccoon',
 'radio-telescope',
 'rainbow',
 'refrigerator',
 'revolver-101',
 'rifle',
 'rotary-phone',
 'roulette-wheel',
 'saddle',
 'saturn',
 'school-bus',
 'scorpion-101',
 'screwdriver',
 'segway',
 'self-propelled-lawn-mower',
 'sextant',
 'sheet-music',
 'skateboard',
 'skunk',
 'skyscraper',
 'smokestack',
 'snail',
 'snake',
 'sneaker',
 'snowmobile',
 'soccer-ball',
 'socks',
 'soda-can',
 'spaghetti',
 'speed-boat',
 'spider',
 'spoon',
 'stained-glass',
 'starfish-101',
 'steering-wheel',
 'stirrups',
 'sunflower-101',
 'superman',
 'sushi',
 'swan',
 'swiss-army-knife',
 'sword',
 'syringe',
 'tambourine',
 'teapot',
 'teddy-bear',
 'teepee',
 'telephone-box',
 'tennis-ball',
 'tennis-court',
 'tennis-racket',
 'theodolite',
 'toaster',
 'tomato',
 'tombstone',
 'top-hat',
 'touring-bike',
 'tower-pisa',
 'traffic-light',
 'treadmill',
 'triceratops',
 'tricycle',
 'trilobite-101',
 'tripod',
 't-shirt',
 'tuning-fork',
 'tweezer',
 'umbrella-101',
 'unicorn',
 'vcr',
 'video-projector',
 'washing-machine',
 'watch-101',
 'waterfall',
 'watermelon',
 'welding-mask',
 'wheelbarrow',
 'windmill',
 'wine-bottle',
 'xylophone',
 'yarmulke',
 'yo-yo',
 'zebra',
 'airplanes-101',
 'car-side-101',
 'faces-easy-101',
 'greyhound',
 'tennis-shoes',
 'toad',
 'clutter')

root='/home/yubo/yubo_tem_code/knockoffnets/data/'
dirs=os.listdir(datasetpath)
dirs.sort()
with open(r'label.txt','w',encoding='utf-8') as f:
    for i in dirs:
        f.write(i)
        f.write('\n')

#print(dirs)
it=0
Matrix = [[] for x in range(257)]                # all filenames under DATA_PATH
for d in dirs:
    for _, _, filename in os.walk(os.path.join(datasetpath,d)):
        for i in filename:
            Matrix[it].append(os.path.join(os.path.join(datasetpath,d),i))  # filename is a list of pic files under the fold
    it = it + 1

#print(Matrix)
with open(os.path.join(root, 'dataset-val.txt'),'w',encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(10):
            f.write(os.path.join(datasetpath,Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')
with open(os.path.join(root, 'dataset-trn.txt'),'w',encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(10,len(Matrix[i])):
            f.write(os.path.join(datasetpath,Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')




def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
#
# mean = [ 0.485, 0.456, 0.406 ]
# std = [ 0.229, 0.224, 0.225 ]

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


train_data = MyDataset(txt=root+'dataset-trn.txt', transform=transform_train)
test_data = MyDataset(txt=root+'dataset-val.txt', transform=transform_test)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
torch.cuda.set_device('cuda:1')

# Model
print('==> Building model..')
#net = ResNet34()
#net = VGG('VGG19')
net = vgg19().cuda()
#net = ResNet18()
# net = PreActResNet18()
#net = googlenet(pretrained = True)
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = models.vgg19(pretrained=True)
#net = SimpleDLA()
if device == 'cuda':
   torch.cuda.set_device('cuda:1')
net = torch.nn.DataParallel(net, device_ids=[1])


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
#
# for idx, (data, target) in enumerate(test_loader):
#     if(idx%10==0):
#         print(str(idx)+' '+str(target))
#
# for idx, (data, target) in enumerate(train_loader):
#     if(idx%10==0):
#         print(str(idx)+' '+str(target))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        with open('train_log.txt', 'a') as f:
            f.write('Epoch: %d | Train: Loss: %.3f | Accuracy: %.3f%% (%d/%d)\n' % (
            epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Test: Loss: %.3f | Accuracy: %.3f%% (%d/%d)' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            with open('test_log.txt', 'a') as f:
                f.write('Epoch: %d | Test: Loss: %.3f | Accuracy: %.3f%% (%d/%d)\n' % (
                    epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

