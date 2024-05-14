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
from torch.utils.data import DataLoader
from collections import defaultdict


import time
import sys


import numpy as np
from knockoff.example_forgetting import order_examples_by_forgetting as forgetting

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from collections import defaultdict as dd
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

#sys.path.append("/home/yubo/yubo_tem_code/knockoffnets")

import knockoff.config as cfg
import torch.nn as nn
#import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo
import knockoff.utils.utils as knockoff_utils
import torchvision.models as torch_models
import torch.nn.functional as F


__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

torch.backends.cudnn.enabled = False

def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


# def soft_cross_entropy(pred, soft_targets, weights=None):
#     soft_targets = soft_targets.to(torch.float32)
#     pred = pred.to(torch.float32)


def stable_softmax(logits):
    max_logits = torch.max(logits, dim=1, keepdim=True).values
    exps = torch.exp(logits - max_logits)
    sum_exps = torch.sum(exps, dim=1, keepdim=True)
    return exps / sum_exps

def soft_cross_entropy(pred, targets, weights=None):
    stable_softmax_probs = stable_softmax(pred)
    log_probs = torch.log(stable_softmax_probs + 1e-6)  # 加小量避免log(0)

    if targets.dim() == 2 and targets.shape[1] == pred.shape[1]:
        soft_targets = targets
    else:
        targets = targets.long()
        soft_targets = F.one_hot(targets, num_classes=pred.size(1)).to(torch.float32)

    if weights is not None:
        device = weights.device
        weights = weights.view(-1, 1).to(device)
        expanded_weights = weights.expand_as(log_probs)
        loss = -torch.sum(soft_targets * log_probs * expanded_weights, dim=1)
    else:
        loss = -torch.sum(soft_targets * log_probs, dim=1)

    return torch.mean(loss)








# def calculate_weights(outputs):
#     confidences, _ = torch.max(outputs, dim=1)
#     weights = 1 - confidences  # 置信度越低，权重越高
#     return weights

def get_all_outputs(model, train_loader, device):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            outputs = model(data)
            all_outputs.append(outputs)
    return torch.cat(all_outputs)


def compute_weights(outputs):
    probabilities = F.softmax(outputs, dim=1)
    confidences, _ = torch.max(probabilities, dim=1)
    weights = torch.exp(-confidences)  # 使用 e^(-confidence)
    return weights


def calculate_weights(forgotten_path, never_forgotten_path, output_json_path):
    # Step 1: Read forgotten counts
    with open(forgotten_path, 'r') as f:
        next(f)  # Skip header
        forgotten_samples = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}

    # Step 2: Identify never forgotten samples
    with open(never_forgotten_path, 'r') as f:
        next(f)  # Skip header
        never_forgotten_samples = {line.strip(): 0 for line in f}  # Assign 0 forget count

    # Combine both dictionaries
    all_samples = {**forgotten_samples, **never_forgotten_samples}

    # Step 3: Calculate weights
    total_forgotten_counts = sum(forgotten_samples.values())
    num_never_forgotten = len(never_forgotten_samples)
    weights = {}

    for sample, count in all_samples.items():
        if count == 0:  # Never forgotten
            weights[sample] = 0.5 / num_never_forgotten
        else:  # Forgotten at least once
            weights[sample] = 0.5 * (count / total_forgotten_counts)

    # Step 4: Save to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(weights, json_file, indent=4)


def train_step(model, train_loader, criterion, optimizer, epoch, device,clip_value=1.0,log_interval=10, writer=None,diag_stats=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # if torch.isnan(inputs).any() or torch.isnan(targets).any():
        #     print(f"Warning: NaN detected in inputs or targets at batch {batch_idx}")
        optimizer.zero_grad()
        outputs = model(inputs)
        # if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        #     print(f"Warning: NaN or Inf detected in model outputs at batch {batch_idx}")
        #     print("Sample outputs:", outputs)
        batch_weights = compute_weights(outputs)
        # if torch.isnan(batch_weights).any():
        #     print(f"Warning: NaN detected in computed weights at batch {batch_idx}")

        batch_weights = batch_weights.to(device)

        # 使用自定义的权重计算函数
        loss = criterion(outputs, targets, weights=batch_weights)
        # if torch.isnan(loss):
        #     print(f"Warning: NaN detected in loss at batch {batch_idx}")

        loss.backward()
        # if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
        #     print(f"Warning: NaN detected in gradients at batch {batch_idx}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        targets = targets.long()
        correct += predicted.eq(targets).sum().item()
        for i, target in enumerate(targets):
            example_id = train_loader.dataset.samples[batch_idx * train_loader.batch_size + i][0]# 假设样本的ID可以通过这种方式获取
            acc = predicted[i].eq(target).item()
            diag_stats[example_id].append((loss.item(), acc))
        # 计算加权损失

        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            acc = 100. * correct / total
            print(f'[Train] Epoch: {epoch:.2f} [{total}/{epoch_size} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {acc:.1f} ({correct}/{total})')

    t_end = time.time()
    sample_ids_to_debug = train_loader.dataset.samples[:5]  # 选择前5个样本进行调试
    print("Debugging sample accuracy changes:")
    for idx, (sample_id, _) in enumerate(sample_ids_to_debug):
        example_stats = diag_stats[sample_id]
        accuracies = [acc for (_, acc) in example_stats]
        print(f"Sample {idx} (ID: {sample_id}): Accuracies: {accuracies}")
    acc = 100. * correct / total
    # 在 train_step 函数的末尾添加


    return train_loss / total, acc



def test_step(model, test_loader, criterion, device, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if len(targets.size()) == 2:
                target_labels = targets.max(1)[1]
            else:
                target_labels = targets
            correct += predicted.eq(target_labels).sum().item()

    t_end = time.time()
    acc = 100. * correct / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc, correct, total))

    if writer and not isinstance(writer, bool):
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc
#
# def analyze_forgetting(diag_stats):
#     forgetting_counts = {}
#     for sample_id, stats in diag_stats.items():
#         # 假设stats中包含了一个列表，每个元素是一个字典，包含了'accuracy'键
#         accuracies = [s['accuracy'] for s in stats]
#         # 计算遗忘次数
#         forgetting_count = sum(1 for i in range(1, len(accuracies)) if accuracies[i-1] == 1 and accuracies[i] == 0)
#         forgetting_counts[sample_id] = forgetting_count
#     return forgetting_counts

# 在训练的适当位置调用该函数


# 输出结果




def train_model(model, trainset, out_path, batch_size=128, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, **kwargs):
    diag_stats = defaultdict(list)  # 用于记录每个样本的诊断信息
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        # outputs = get_all_outputs(model, train_loader, device)
        # weights = calculate_weights(outputs)
        train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,log_interval=log_interval, diag_stats=diag_stats)
        #test_loss, test_acc = test_step(model, test_loader, criterion_train, device, epoch=epoch)

        # train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
        #                                    log_interval=log_interval)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned = forgetting.compute_forgetting_statistics(
            diag_stats, epochs)

        ordered_examples, ordered_values = forgetting.sort_examples_by_forgetting(unlearned_per_presentation, epochs)
        # 在 train_model 函数中，在调用 sort_examples_by_forgetting 之前添加
        sorted_forgetting = sorted(unlearned_per_presentation.items(), key=lambda x: len(x[1]), reverse=True)

        # 写入文件
        with open(os.path.join(out_path, 'forgotten_examples_DBplusFilterCaltech.txt'), 'w') as file_forgotten, \
                open(os.path.join(out_path, 'never_forgotten_examples_DBplusFilterCaltech.txt'), 'w') as file_never_forgotten:
            file_forgotten.write("Example ID\tTotal Forgetting Count\n")
            file_never_forgotten.write("Example ID\n")

            for example_id, forgetting_events in sorted_forgetting:
                forgetting_count = len(forgetting_events)
                if forgetting_count > 0:
                    file_forgotten.write(f"{example_id}\t{forgetting_count}\n")
                else:
                    file_never_forgotten.write(f"{example_id}\n")
        if test_loader is not None:
            test_loss, test_acc = test_step(model, test_loader, criterion_train, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)
        # forgetting_counts = analyze_forgetting(diag_stats)
        # for sample_id, count in forgetting_counts.items():
        #     print(f"Sample {sample_id}: Forgot {count} times")


        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            # af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    # 在 train_model 函数内部，在 return model 之前添加
    # # 汇总并分析整个训练过程中的遗忘事件
    # presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned = forgetting.compute_forgetting_statistics(
    #     diag_stats, epochs)
    # ordered_examples, ordered_values = forgetting.sort_examples_by_forgetting([unlearned_per_presentation],
    #                                                                           [first_learned], epochs)
    # print("Overall forgotten examples ranking:")
    # for idx, example_id in enumerate(ordered_examples[:10]):
    #     print(f"Example ID: {example_id}, Total Forgetting Count: {ordered_values[idx]}")
    #
    # return model

    # 在 train_model 函数的循环结束后

    print(f"Training complete. Best test accuracy: {best_test_acc}%")  # 训练结束后输出最大测试准确率
    return model

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
        directory = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/DBplusFilterCaltech"
        #directory ="/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/afterFilterCaltechImages"# 修改为你的图片目录
        #directory = "/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/linearinterpolationExpand"

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

    transfer_parts_dir = osp.join(params['model_dir'], 'transferset_parts_DBplusFilterCaltech')
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
        criterion_train = soft_cross_entropy
        train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
    # 计算权重并保存到JSON的逻辑应该放在这里，确保它是在train_model函数执行完之后进行的。
    #forgotten_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusRealRandom/forgotten_examples_DBplusfilterCaltechImages.txt'  # 替换为实际路径
    forgotten_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusFilterCaltech-20/forgotten_examples_DBplusFilterCaltech.txt'
    never_forgotten_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusFilterCaltech-20/never_forgotten_examples_DBplusFilterCaltech.txt'  # 替换为实际路径
    output_json_path = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/models/adversary/cifar10-vgg19-DBplusFilterCaltech-20/output_weights_DBplusFilterCaltech.json'  # 指定输出JSON文件的路径
    # 调用函数计算权重并保存到JSON
    calculate_weights(forgotten_path, never_forgotten_path, output_json_path)

