#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torchvision.transforms.functional import to_tensor

import sys

#sys.path.append("/home/yubo/yubo_tem_code/knockoffnets")

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo


__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

torch.backends.cudnn.enabled = False



class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    # IMG_EXTENSIONS = ('.jpg','.jpeg','png','.ppm','.bmp','.pgm,','.tif')

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    #     self.extensions = IMG_EXTENSIONS
    #     self.samples = samples
    #     self.targets = [s[1] for s in samples]
    #     self.transform = transform
    #     self.target_transform = target_transform
    #
    def __getitem__(self, index):
        # path = self.samples[index][0]
        # target = self.samples[0][index]
        directory = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/stds0.5_caltech0.7_0.5"  # 修改为你的图片目录
        filename,target = self.samples[index]
        full_path = os.path.join(directory, filename)
        #print(os.path.abspath(full_path))
        #print(f"Attempting to load image at: {path}")  # Debugging line
        # with open('new_Image_134_23150_truck.png', 'rb') as f:
        #     print("File can be opened.")
        #print(os.path.abspath('new_Image_134_23150_truck.png'))
        # img = self.loader(path)
        #path = self.imgs[index]
        img = self.loader(full_path)
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    #
    # def __len__(self):
    #     return len(self.samples)
    #
    # def loader(self,path):
    #     with open(path,'rb') as f:
    #         img = Image.open(f)
    #         return img.convert('RGB')


class TransferSetImages(Dataset):
    def __init__(self, samples: object, transform: object = None, target_transform: object = None) -> object:
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(
            sample_x)))  # assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def get_transferset_from_parts(directory):
    all_parts = sorted([os.path.join(directory, fname) for fname in os.listdir(directory) if "part_" in fname])
    combined_transferset = []

    for part_path in all_parts:
        with open(part_path, 'rb') as pf:
            part_data = pickle.load(pf)
            print(f"Loaded {len(part_data)} samples from {part_path}")  # 打印每个部分的长度
            combined_transferset.extend(part_data)

    print(f"Total samples in combined transferset: {len(combined_transferset)}")  # 打印合并后的总长度
    return combined_transferset



def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']

    # ----------- Set up transferset
    # transferset_path = osp.join(model_dir, 'transferset.pickle')
    # print(f"Model directory is: {model_dir}")
    # print(f"Transferset path is: {transferset_path}")
    # with open(transferset_path, 'rb') as rf:
    #     transferset_samples = pickle.load(rf)
    # num_classes = transferset_samples[0][1].size(0)
    # print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    transfer_parts_dir = osp.join(params['model_dir'], 'transferset_parts_stds0.5_caltech0.7_0.5')
    if osp.exists(transfer_parts_dir):
        transferset_samples = get_transferset_from_parts(transfer_parts_dir)
    else:
        transferset_path = osp.join(params['model_dir'], 'transferset.pickle')
        print("transferset_path:", transferset_path)
        if osp.exists(transferset_path):
            print("File exists. Attempting to load transferset.")
            with open(transferset_path, 'rb') as rf:
                transferset_samples = pickle.load(rf)
        else:
            print("No valid transferset found.")

        with open(transferset_path, 'rb') as rf:
            transferset_samples = pickle.load(rf)
    if not transferset_samples or len(transferset_samples) == 0:
        raise ValueError('transferset_samples is empty')
    print('First sample in transferset_samples:', transferset_samples[0])

    num_classes = transferset_samples[0][1].size(0)
    print('Number of classes:', num_classes)

    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # # ----------- Clean up transfer (if necessary)
    # if params['argmaxed']:
    #     new_transferset_samples = []
    #     print('=> Using argmax labels (instead of posterior probabilities)')
    #     for i in range(len(transferset_samples)):
    #         x_i, y_i = transferset_samples[i]
    #         argmax_k = y_i.argmax()
    #         y_i_1hot = torch.zeros_like(y_i)
    #         y_i_1hot[argmax_k] = 1.
    #         new_transferset_samples.append((x_i, y_i_1hot))
    #     transferset_samples = new_transferset_samples

    # ----------- Set up testset



    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    model = model.to(device)

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        transferset = samples_to_transferset(transferset_samples, budget=b, transform=transform)
        print()
        print('=> Training at budget = {}'.format(len(transferset)))

        optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], **params)
        print(params)

        checkpoint_suffix = '.{}'.format(b)
        criterion_train = model_utils.soft_cross_entropy
        model_utils.train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
