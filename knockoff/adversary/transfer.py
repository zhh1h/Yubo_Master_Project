#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import sys
import os.path as osp
import os
from torchvision.datasets import VisionDataset
from PIL import Image
import tracemalloc
#from adaptive import AdaptiveAdversary
from knockoff.models.cifar import vgg19
from sklearn.cluster import MiniBatchKMeans
import shutil  # 导入 shutil 模块



sys.path.append("/home/yubo/yubo_tem_code/knockoffnets")

import pickle
import json
from datetime import datetime

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn


from knockoff import datasets

import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


#torch.load('',map_location='cpu')
torch.nn.Module.dump_patches = True



class FlatDirectoryImageDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(FlatDirectoryImageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.image_files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.data = self.image_files  # 或者其他适当的数据结构

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.root, img_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.image_files)

tracemalloc.start()

class RandomAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []



    def get_transferset(self, budget, transfer_out_dir):
        print("Running randomAdversary get_transferset")
        start_B = 0
        end_B = budget
        total_saved = 0
        part_num = 0

        # 确保输出目录存在
        if not os.path.exists(transfer_out_dir):
            os.makedirs(transfer_out_dir)

        with tqdm(total=budget) as pbar:
            while total_saved < budget:
                idxs = np.random.choice(list(self.idx_set), replace=False,
                                        size=min(self.batch_size, budget - total_saved))
                self.idx_set = self.idx_set - set(idxs)

                if len(self.idx_set) == 0:
                    print('=> Query set exhausted. Now repeating input examples.')
                    self.idx_set = set(range(len(self.queryset)))

                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                y_t = self.blackbox(x_t).cpu()

                if hasattr(self.queryset, 'samples'):
                    img_t = [self.queryset.samples[i][0] for i in idxs]
                else:
                    img_t = [self.queryset.data[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                part_transferset = []
                for i in range(x_t.size(0)):
                    img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    part_transferset.append((img_t_i, y_t[i].cpu().squeeze()))

                # 将这部分的transferset保存为一个新的pickle文件
                part_path = os.path.join(transfer_out_dir, f"part_{part_num}.pickle")
                with open(part_path, 'wb') as pf:
                    pickle.dump(part_transferset, pf)

                total_saved += len(idxs)
                part_num += 1
                pbar.update(len(idxs))

        return total_saved
class AdaptiveAdversary(object):
    def __init__(self, blackbox, queryset, out_path, batch_size=8, num_workers=15, flush_interval=1000, num_clusters=10):
        self.blackbox = blackbox
        self.queryset = queryset
        self.out_path = out_path
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_clusters = num_clusters
        self.idx_set = set(range(len(queryset)))
        self.transferset = []

        # 初始化自定义的 VGG19 模型
        self.feature_extractor = vgg19()  # 或 vgg19_bn() 如果使用带批量归一化的版本
        self.feature_extractor.eval()  # 设置为评估模式

        # 预处理数据集以获得聚类标签
        self.cluster_labels = self.preprocess_dataset(queryset)
        self.samples_per_cluster = {i: [] for i in range(num_clusters)}



    #     self._restart()
    #
    # def _restart(self):
    #     np.random.seed(cfg.DEFAULT_SEED)
    #     torch.manual_seed(cfg.DEFAULT_SEED)
    #     torch.cuda.manual_seed(cfg.DEFAULT_SEED)
    #
    #     self.idx_set = set(range(len(self.queryset)))
    #     self.transferset = []

    def preprocess_dataset(self, queryset):
        # 使用批量处理进行特征提取
        batch_size = 128  # 可以调整批次大小
        all_features = []
        for i in range(0, len(queryset), batch_size):
            batch = torch.stack([queryset[j][0] for j in range(i, min(i + batch_size, len(queryset)))])
            features = self.extract_features(batch)
            all_features.extend(features.cpu().numpy())

        all_features = np.stack(all_features)
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0).fit(all_features)
        return kmeans.labels_

    def extract_features(self, x):
        # 假设 x 是已经加载的图像数据（Tensor对象）
        # 如果它不是 Tensor 或不是预期的形状，您可能需要先进行转换或预处理
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected input to be a Tensor.")

        # 确保 x 是 4D (batch_size, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加 batch 维度

        # 使用 VGG19 的卷积层提取特征
        with torch.no_grad():
            features = self.feature_extractor.features(x)
            features = features.view(features.size(0), -1)
        return features



    def evaluate_sample(self, y_t):
        # 根据模型的响应评估每个样本的置信度
        confidence_scores = y_t.max(dim=1).values
        return confidence_scores

    def update_sampling_strategy(self, idxs, confidence_scores, confidence_threshold, cluster_sample_limit):
        for idx, conf in zip(idxs, confidence_scores):
            cluster = self.cluster_labels[idx]
            if conf < confidence_threshold and len(self.samples_per_cluster[cluster]) < cluster_sample_limit:
                self.samples_per_cluster[cluster].append(idx)
                if idx in self.idx_set:
                    self.idx_set.remove(idx)
    def save_progress(self, progress_path, part_num, actual_saved):
        with open(progress_path, 'wb') as pf:
            pickle.dump({
                'part_num': part_num,
                'actual_saved': actual_saved,
                'idx_set': self.idx_set,
                'samples_per_cluster': self.samples_per_cluster
            }, pf)

    def load_progress(self, progress_path):
        if os.path.exists(progress_path):
            with open(progress_path, 'rb') as pf:
                progress = pickle.load(pf)
                self.idx_set = progress['idx_set']
                self.samples_per_cluster = progress['samples_per_cluster']
                print(
                    f"Resuming from part {progress['part_num']}, with {progress['actual_saved']} samples already saved.")

                return progress['part_num'], progress['actual_saved']
        return 0, 0

    def get_transferset(self, budget, out_path,images_out_dir,progress_out_path):
        print("Running adaptiveAdversary get_transferset")
        start_B = 0
        end_B = budget
        total_saved = 0
        part_num = 0
        actual_saved = 0

        # 定义抽样阈值
        confidence_threshold = 0.7
        cluster_sample_limit = len(self.queryset) // self.num_clusters
        part_num, actual_saved = self.load_progress(progress_out_path)
        total_saved = actual_saved

        # 确保输出目录存在
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(images_out_dir):
            os.makedirs(images_out_dir)

        # 重置进度
        self.idx_set = set(range(len(self.queryset)))
        self.samples_per_cluster = {i: [] for i in range(self.num_clusters)}

        with tqdm(total=budget,initial=total_saved) as pbar:
            while total_saved < budget:
                if len(self.idx_set) == 0:
                    break

                selected_clusters = np.random.choice(range(self.num_clusters), size=self.batch_size, replace=True)
                idxs = []
                for cluster in selected_clusters:
                    cluster_idxs = [i for i in self.idx_set if self.cluster_labels[i] == cluster]
                    if cluster_idxs:
                        idx = np.random.choice(cluster_idxs)
                        idxs.append(idx)
                        self.idx_set.remove(idx)

                # 检查 queryset 是否有 'image_files' 属性
                if hasattr(self.queryset, 'image_files'):
                    img_t = [self.queryset.image_files[i] for i in idxs]
                else:
                    # 如果没有 'image_files' 属性，这里需要根据您的数据集结构进行调整
                    raise AttributeError("'FlatDirectoryImageDataset' object has no attribute 'image_files'")

                x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)

                try:
                    y_t = self.blackbox(x_t).cpu()
                except Exception as e:
                    print(f"Exception during blackbox prediction: {e}")
                    continue

                confidence_scores = self.evaluate_sample(y_t)
                part_transferset = []
                valid_indices = []
                valid_indices = []
                for i, (idx, conf_score) in enumerate(zip(idxs, confidence_scores)):
                    cluster_label = self.cluster_labels[idx]
                    if conf_score < confidence_threshold and len(
                            self.samples_per_cluster[cluster_label]) < cluster_sample_limit:
                        valid_indices.append(i)  # 保存有效样本的索引
                        print(
                            f"Selected: Sample index: {idx}, Cluster: {cluster_label}, Confidence Score: {conf_score:.4f}")
                    else:
                        print(
                            f"Skipped: Sample index: {idx}, Cluster: {cluster_label}, Confidence Score: {conf_score:.4f}")

                # 只保存符合条件的样本
                for i in valid_indices:
                    part_transferset.append((img_t[i], y_t[i].cpu().squeeze()))
                    img_full_path = os.path.join(self.queryset.root, img_t[i])
                    # 构建目标路径
                    dest_img_path = os.path.join(images_out_dir, os.path.basename(img_t[i]))
                    # 复制图片
                    shutil.copy(img_full_path, dest_img_path)
                # 保存当前部分的transferset
                part_path = os.path.join(out_path, f"part_{part_num}.pickle")
                with open(part_path, 'wb') as pf:
                    pickle.dump(part_transferset, pf)

                part_saved = len(part_transferset)
                actual_saved += part_saved  # 累加实际保存的样本数量
                if part_saved > 0:
                    self.save_progress(progress_out_path, part_num, actual_saved)

                    print(f"Part {part_num} saved, {part_saved} samples.")
                if os.path.exists(part_path) and os.path.getsize(part_path) > 0:
                    print(f"Part {part_num} saved successfully, file size: {os.path.getsize(part_path)} bytes.")
                else:
                    print(f"Warning: Saving part {part_num} failed or file is empty.")

                total_saved += len(idxs)
                part_num += 1
                pbar.update(len(idxs))
        print(f"Total expected samples: {total_saved}")
        print(f"Total actual samples saved: {actual_saved}")  # 打印实际保存的样本数量
        return total_saved


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                        choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('--custom_query_path', type=str, help='Path to custom query dataset', default=None)

    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)
    # parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
    #                     default=None)
    # parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
    #                     default=None)
    # parser.add_argument('--tau_data', metavar='N', type=float, help='Frac. of data to sample from Adv data',
    #                     default=1.0)
    # parser.add_argument('--tau_classes', metavar='N', type=float, help='Frac. of classes to sample from Adv data',
    #                     default=1.0)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    base_out_path = params['out_dir']
    knockoff_utils.create_dir(base_out_path)
    progress_out_path = os.path.join(base_out_path, 'progress.pkl')

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # device = torch.device('cpu')
    # print(f'!!!!!!!!!!!!!!!!!!!!{torch.cuda.is_available()}!!!!!!!!!!!!!!!!!!!!!')
    # ----------- Set up queryset
    # queryset_name = params['queryset']
    # valid_datasets = datasets.__dict__.keys()
    # if queryset_name not in valid_datasets:
    #     raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    # modelfamily = datasets.dataset_to_modelfamily[queryset_name] if params['modelfamily'] is None else params['modelfamily']
    # transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    # if queryset_name == 'ImageFolder':
    #     assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
    #     queryset = datasets.__dict__[queryset_name](root=params['root'], transform=transform)
    # else:
    #     queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    if params['custom_query_path']:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        queryset = FlatDirectoryImageDataset(root=params['custom_query_path'], transform=transform)
    else:
        queryset_name = params['queryset']
        valid_datasets = datasets.__dict__.keys()
        if queryset_name not in valid_datasets:
            raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
        modelfamily = datasets.dataset_to_modelfamily[queryset_name] if params['modelfamily'] is None else params[
            'modelfamily']
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    # #transfer_out_path = osp.join(out_path, 'transferset.pickle')
    # transfer_out_dir = osp.join(out_path, 'transferset_parts_0.7_0.5_DB_std')
    # if params['policy'] == 'random':
    #     adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)
    # elif params['policy'] == 'adaptive':
    #     # 创建一个 AdaptiveAdversary 实例
    #     adversary = AdaptiveAdversary(
    #         blackbox=blackbox,
    #         queryset=queryset,
    #         out_path=params.get('out_path', 'path_to_save_results'),
    #         batch_size=batch_size,
    #         num_workers=params.get('num_workers', 15),
    #         flush_interval=params.get('flush_interval', 1000),
    #         num_clusters=params.get('num_clusters', 10)
    #     )
    # else:
    #     raise ValueError("Unrecognized policy")
    final_out_path = osp.join(base_out_path, 'transferset_parts_epsilonExpand40random' if params[
                                                                                       'policy'] == 'random' else 'transfer_std_s0.7')
    knockoff_utils.create_dir(final_out_path)

    #final_out_path = None
    if params['policy'] == 'random':
        adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)
    elif params['policy'] == 'adaptive':
        adversary = AdaptiveAdversary(
            blackbox=blackbox,
            queryset=queryset,
            out_path=final_out_path,
            batch_size=batch_size,
            num_workers=params.get('num_workers', 15),
            flush_interval=params.get('flush_interval', 1000),
            num_clusters=params.get('num_clusters', 10)
        )
    else:
        raise ValueError("Unrecognized policy")

    #     print('=> constructing transfer set...')
#     transferset = adversary.get_transferset(params['budget'])
#     with open(transfer_out_path, 'wb') as wf:
#         pickle.dump(transferset, wf)
#     print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))
#
#     # Store arguments
#     params['created_on'] = str(datetime.now())
#     params_out_path = osp.join(out_path, 'params_transfer.json')
#     with open(params_out_path, 'w') as jf:
#         json.dump(params, jf, indent=True)
#
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')  # 按行号对统计信息进行排序
#
# # 打印内存使用情况的前10行
# for stat in top_stats[:10]:
#     print(stat)
#
#
    tracemalloc.stop()
#     print('=> constructing transfer set...')
#     total_saved = adversary.get_transferset(params['budget'], transfer_out_dir)
#     print('=> total {} samples saved in parts under: {}'.format(total_saved, transfer_out_dir))
#     # total_saved = adversary.get_transferset(params['budget'], out_path)
#     # print('=> total {} samples saved in parts under: {}'.format(total_saved, out_path))
#
#     params['created_on'] = str(datetime.now())
#     params_out_path = osp.join(out_path, 'params_transfer.json')
#     with open(params_out_path, 'w') as jf:
#         json.dump(params, jf, indent=True)
#     print('=> constructing transfer set...')
#     if params['policy'] == 'random':
#         total_saved = adversary.get_transferset(params['budget'], final_out_path)
#         print('=> total {} samples saved in parts under: {}'.format(total_saved, final_out_path))
#     elif params['policy'] == 'adaptive':
#         total_saved = AdaptiveAdversary.get_transferset(params['budget'], final_out_path)
#         print('=> total {} samples saved in parts under: {}'.format(total_saved, final_out_path))
    print('=> constructing transfer set...')
    if params['policy'] == 'random':
        total_saved = adversary.get_transferset(params['budget'], final_out_path)
    if params['policy'] == 'adaptive':
        images_out_dir = '/home/yubo/PycharmProjects/Yubo_Master_Project_Remote/num_gradient/num_gradient_descent_master/std_s0.7'
        total_saved = adversary.get_transferset(params['budget'], final_out_path, images_out_dir,progress_out_path)
    print('=> total {} samples saved in parts under: {}'.format(total_saved, final_out_path))

    # 保存转移集构建过程的参数
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(final_out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
