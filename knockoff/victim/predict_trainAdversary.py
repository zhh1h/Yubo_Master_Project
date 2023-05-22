import json
import os

import torch
from PIL import Image
from torchvision import transforms
import os.path as osp
import random

from knockoff.victim.blackbox import Blackbox
import argparse
import knockoff.models.zoo as zoo
import knockoff.utils.model as model_utils
import numpy as np
import knockoff.config as cfg
from knockoff import datasets
from knockoff.adversary.train import samples_to_transferset
from knockoff.adversary.train import get_optimizer

from datetime import datetime


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
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
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
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    args = parser.parse_args()
    params = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = '../data/random_images_random/'
    #'../data/random_images_200-255/'
    #"../data/random_images"
    #'../data/random_images_random/'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root)[0:int(params['budgets'])] if i.endswith(".jpg")]

    # read class_indict
    json_path = '../models/victim/caltech256-resnet34/params.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create random_victim_model
    blackbox_random = Blackbox.from_modeldir('../models/victim/caltech256-resnet34', device)
    random_victim_model = blackbox_random.get_model()

    # load random_victim_model weights
    weights_path = "../models/victim/caltech256-resnet34/checkpoint.pth.tar"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    random_victim_model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    random_victim_model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    samples = []
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            path_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)
                path_list.append(img_path)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = random_victim_model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)

            # samples = [(img_path,predict)for image_path, score in zip(img_path,predict)]

            #for idx, score in enumerate(predict):
            # #
                 #print(f"image: {img_path_list[ids * batch_size + idx]}, \nscore: {score}, \nscoreLen: {len(score)}")
            # # samples = [(each_idx, each_score) for each_idx, each_score in zip(img_list,predict.tolist())]
            for path, score in zip(path_list, predict.tolist()):
                sample = [(path, torch.tensor(score))]
                samples.extend(sample)




    # 处理最后一批不足 batch_size 张图片的情况
    last_batch_size = len(img_path_list) % batch_size
    if last_batch_size > 0:
        img_list = []
        path_list = []
        for img_path in img_path_list[-last_batch_size:]:
            assert os.path.exists(img_path), f"file:'{img_path}' dose not exist."
            img = Image.open(img_path)
            img = data_transform(img)
            img_list.append(img)
            path_list.append(img_path)

        # batch_img = torch.stack(img_list, dim=0)
        output = random_victim_model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)

        for path, score in zip(path_list, predict.tolist()):
            sample = [(path, torch.tensor(score))]
            samples.extend(sample)

            num_classes = samples[0][1].size(0)

        #for idx, score in enumerate(predict):
        #
            #print(f"image: {img_path_list[ids * batch_size + idx]}, \nscore: {score}, \nscoreLen: {len(score)}")
    for path,score in samples:
        class_index = torch.argmax(score)
        #print(f"image:{path},\nscore:{score}, \nscorelen:{len(score)},\nclass:{class_index}")
        print(f"class:{class_index}")





    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, 'imagenet', pretrained, num_classes=256)
    model = model.to(device)

    # ----------- Set up testset
    # dataset_name = params['testdataset']
    # valid_datasets = datasets.__dict__.keys()
    # modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    # transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    # if dataset_name not in valid_datasets:
    #     raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    # dataset = datasets.__dict__[dataset_name]
    # testset = dataset(train=False, transform=transform)
    # if len(testset.classes) != num_classes:
    #     raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Train

    budgets = [int(b) for b in params['budgets'].split(',')]
    transform = datasets.modelfamily_to_transforms['imagenet']['train']

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        # random_set = samples_to_transferset(samples, budget=b, transform=transform, target_transform=None)
        print()
        print('=> Training at budget = {}'.format(len(samples)))

        optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], **params)
        print(params)

        checkpoint_suffix = '.{}'.format(b)
        criterion_train = model_utils.soft_cross_entropy
        model_dir = '../models/victim/random_images_adversary_random'

        def train_and_test(model, random_set, k, model_dir, criterion_train, checkpoint_suffix='', device='cpu',
                           optimizer=None,
                           **params):
            n = len(random_set)
            testset = []
            trainset = []
            testset_index = []
            trainset_index = list(range(0, n))
            while len(testset_index) < n // k:
                r = random.randint(0, n - 1)
                if r in trainset_index:
                    trainset_index.remove(r)
                    testset_index.append(r)
            for i in testset_index:
                testset.append(random_set[i])
            for i in trainset_index:
                trainset.append(random_set[i])

            trainset = samples_to_transferset(trainset, budget=int(b / k * (k - 1)), transform=transform,
                                              target_transform=None)
            testset = samples_to_transferset(testset, budget=int(b / k), transform=transform, target_transform=None)
            model_utils.train_model(model, trainset, model_dir, testset=testset, criterion_train=criterion_train,
                                    checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)


        k = 5

        for f in range(k):
            print(f'-------------------------------start fold:{f}-----------------------------------------')

            train_and_test(model=model, random_set=samples, k=5, model_dir=model_dir, criterion_train=criterion_train,
                       checkpoint_suffix='', device=device, optimizer=None)

            print(f'-------------------------------end fold:{f}-----------------------------------------')

        # testset = dataset(train=True, transform=transform)
        # n = len(random_set)
        # k = 5
        # testset = []
        # trainset = []
        # testset_index = []
        # trainset_index = list(range(0, n))
        # while len(testset_index) < n / k:
        #     r = random.randint(0, n)
        #     if trainset_index.__contains__(r):
        #         trainset_index.remove(r)
        #         testset_index.append(r)
        # for i in testset_index,trainset_index:
        #     testset.append(random_set[i])
        #     trainset.append(random_set[i])
        #     model_utils.train_model(model, trainset, model_dir, testset=testset, criterion_train=criterion_train,checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
