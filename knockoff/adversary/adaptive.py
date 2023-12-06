#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os

import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


import torchvision.transforms as transforms
from PIL import Image
from knockoff.models.cifar import vgg19
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


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
        batch_size = 64  # 可以调整批次大小
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


    def update_sampling_strategy(self, idxs, confidence_scores):
        # 根据置信度和多样性更新抽样策略
        confidence_threshold = 0.7
        some_cluster_sample_threshold = len(self.queryset) / self.num_clusters
        for idx, conf in zip(idxs, confidence_scores):
            cluster = self.cluster_labels[idx]
            if conf < confidence_threshold or len(self.samples_per_cluster[cluster]) < some_cluster_sample_threshold:
                self.samples_per_cluster[cluster].append(idx)
            else:
                self.idx_set.remove(idx)

    def get_transferset(self, budget, out_path):
        print("Running randomAdversary get_transferset")
        start_B = 0
        end_B = budget
        total_saved = 0
        part_num = 0

        # 确保输出目录存在
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with tqdm(total=budget) as pbar:
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
                self.update_sampling_strategy(idxs, confidence_scores)

                part_transferset = []
                for i in range(x_t.size(0)):
                    self.transferset.append((img_t[i], y_t[i].squeeze()))

                # 保存当前部分的transferset
                part_path = os.path.join(out_path, f"part_{part_num}.pickle")
                with open(part_path, 'wb') as pf:
                    pickle.dump(part_transferset, pf)

                total_saved += len(idxs)
                part_num += 1
                pbar.update(len(idxs))

        return total_saved



