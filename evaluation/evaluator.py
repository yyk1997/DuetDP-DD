import torch
from PIL import Image
import copy
import numpy as np
from scipy import linalg
from scipy.stats import entropy
import random
import logging

from pytorch_fid.inception import fid_inception_v3
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FLD import FLD
from fld.metrics.PrecisionRecall import PrecisionRecall
from fld.metrics.FID import FID
import ImageReward as RM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import logging

from evaluation.ema import ExponentialMovingAverage
from evaluation.classifier.wrn import WideResNet
from evaluation.classifier.resnet import ResNet
from evaluation.classifier.resnext import ResNeXt
import evaluation.classifier.convnet as CN
import evaluation.classifier.resnet_18 as resnet_18
import evaluation.classifier.dense121 as dense121

from torch.optim.lr_scheduler import LambdaLR

import math
import copy
import os

from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from pathlib import Path


def extract_logits_features_batched(
    model,
    images_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
):
    """
    images_cpu: [N, C, H, W], kept on CPU
    return:
        probs_cpu:    [N, num_classes]
        features_cpu: [N, feat_dim]
        pred_cpu:     [N]
    """
    probs_list = []
    features_list = []
    pred_list = []

    model.eval()
    with torch.no_grad():
        for start in range(0, images_cpu.size(0), batch_size):
            end = min(start + batch_size, images_cpu.size(0))
            batch = images_cpu[start:end].to(device, non_blocking=True)

            logits, features = model(batch * 2 - 1, return_features=True)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            probs_list.append(probs.cpu())
            features_list.append(features.cpu())
            pred_list.append(pred.cpu())

            del batch, logits, features, probs, pred
            torch.cuda.empty_cache()

    probs_cpu = torch.cat(probs_list, dim=0)
    features_cpu = torch.cat(features_list, dim=0)
    pred_cpu = torch.cat(pred_list, dim=0)

    return probs_cpu, features_cpu, pred_cpu



class PublicDistillationDataset(Dataset):
    def __init__(self, selected_public_distillations, transform=None):
        """
        Args:
            selected_public_distillations:
                List[List[np.ndarray]], outer index = class label
            transform:
                torchvision transforms applied to image
        """
        self.samples = []
        self.labels = []
        self.transform = transform

        for class_idx, class_samples in enumerate(selected_public_distillations):
            for img in class_samples:
                self.samples.append(img)
                self.labels.append(class_idx)

        self.samples = np.asarray(self.samples) # [N,H,W,C]
        self.labels = np.asarray(self.labels) # [N]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]  # np.ndarray, uint8, HWC
        label = self.labels[idx]

        # numpy -> PIL Image
        img = Image.fromarray(img)

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback: ToTensor
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img, label


def kmeans_sampling(features, k, random_state=None):
    """
    K-Means Sampling: 选取离 K-Means 聚类中心最近的样本 (Prototypical Sampling)

    Args:
        features: [N, D] torch tensor (可以是 CUDA tensor)
        k: 需要选取的样本数量
        random_state: 固定随机种子，保证实验可复现性

    return: selected indices (list)
    """
    # 1. 预处理：K-Means 通常在 CPU 上用 sklearn 跑比较稳定
    # 如果 features 在 GPU 上，需要先转到 CPU 并转为 numpy
    features_np = features.detach().cpu().numpy()

    # 2. 执行 K-Means 聚类
    # n_init=10 是默认值，但在做科研时显式指定比较好，防止陷于局部最优
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(features_np)

    # 获取 K 个虚拟质心 (Cluster Centroids) -> [k, D]
    centers = torch.from_numpy(kmeans.cluster_centers_).to(features.device)

    # 3. 寻找离质心最近的真实样本 (Mapping to nearest neighbors)
    # 计算所有样本到所有质心的距离矩阵
    # features: [N, D], centers: [k, D] -> dist_matrix: [N, k]
    dist_matrix = torch.cdist(features, centers)

    # 4. 选取索引
    # 对每一列（每个质心），找到距离最小的那个样本的行索引
    # argmin(dim=0) 返回 [k] 大小的 tensor
    selected_indices = torch.argmin(dist_matrix, dim=0).tolist()

    # 注意：理论上极小概率会出现同一个样本离两个质心都最近（由于 K-Means 机制，通常不会）
    # 如果为了绝对严谨，可以加一步去重，但在 Coreset 筛选中通常不做处理

    return selected_indices



def random_indices(y, nclass=10, intraclass=False, device='cuda:0'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(data, target, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    target_a, target_b = target, target[index]
    return mixed_data, target_a, target_b, lam



def set_global_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups


def k_center_greedy(features, k, use_norm = True):
    """
    features: [N, D] torch tensor
    return: selected indices (list)
    """
    N, D = features.shape
    device = features.device

    if use_norm:
        # 进行 L2 归一化，使得度量等价于余弦距离
        features = torch.nn.functional.normalize(features, p=2, dim=1)

    # 随机选择第一个中心点
    start_idx = torch.randint(0, N, (1,)).item()
    centers = [start_idx]

    # 初始化每个点到当前中心点集合的最小距离
    # 使用 float('inf') 确保第一次更新正确
    min_distances = torch.full((N,), float('inf'), device=device)

    # 第一次更新：所有点到第一个中心点的距离
    # 这里不需要计算全矩阵，只计算 [N, D] 和 [1, D] 的距离
    current_center_feat = features[start_idx].unsqueeze(0)
    min_distances = torch.cdist(current_center_feat, features).squeeze(0)

    for _ in range(1, k):
        # 1. 选出距离当前所有中心点最远的点 (Max-Min 策略)
        new_center_idx = torch.argmax(min_distances).item()
        centers.append(new_center_idx)

        # 2. 计算所有点到这个新中心点的距离
        new_center_feat = features[new_center_idx].unsqueeze(0)
        dist_to_new_center = torch.cdist(new_center_feat, features).squeeze(0)

        # 3. 更新最小距离向量：min(之前的最小距离, 到新中心的距离)
        min_distances = torch.minimum(min_distances, dist_to_new_center)

    return centers

def k_center_greedy_with_one_init(
    features: torch.Tensor,
    k: int,
    init_idx,
    metric: str = "euclidean",
    use_norm = True
):
    """
    features: [N, D] torch tensor (建议先 L2 normalize)
    k: number of centers
    init_idx: 指定第一个中心点索引（None 表示随机）
    """
    device = features.device
    N = features.size(0)

    if use_norm:
        # 进行 L2 归一化，使得度量等价于余弦距离
        features = torch.nn.functional.normalize(features, p=2, dim=1)

    # if init_idx is None:
    #     init_idx = torch.randint(0, N, (1,), device=device).item()

    centers = []

    if init_idx is None:
        # 如果没有指定，随机选择一个作为起始
        rand_idx = torch.randint(0, N, (1,), device=device).item()
        centers = [rand_idx]
    elif isinstance(init_idx, int):
        centers = [init_idx]
    else:
        # 如果是 list 或 tensor，转换为 list
        if isinstance(init_idx, torch.Tensor):
            centers = init_idx.cpu().tolist()
        else:
            centers = list(init_idx)

    # 检查初始点数量是否已经超过 k
    if len(centers) > k:
        print(f"Warning: init_idx length ({len(centers)}) is larger than k ({k}). Returning init_idx.")
        return centers[:k]

    # 我们将其初始化为无穷大，这样第一次 update 就会直接取距离
    min_dist = torch.full((N,), float('inf'), device=device)

    # 辅助函数：计算所有点到指定 center_idx 的距离
    def compute_dist(center_idx):
        if metric == "euclidean":
            # dist = ||x - y||
            return torch.norm(features - features[center_idx], dim=1)
        elif metric == "cosine":
            # dist = 1 - cos(x, y)
            return 1 - torch.matmul(features, features[center_idx])
        else:
            raise ValueError(f"Unknown metric: {metric}")

    for center_idx in centers:
        dist_new = compute_dist(center_idx)
        min_dist = torch.minimum(min_dist, dist_new)

    # 只需要循环 k - len(centers) 次
    num_needed = k - len(centers)

    for _ in range(num_needed):
        # 找到距离当前中心集合最远的点 (ArgMax MinDist)
        idx = torch.argmax(min_dist).item()
        centers.append(idx)

        # 计算新选点到所有点的距离，并更新 min_dist
        dist_new = compute_dist(idx)
        min_dist = torch.minimum(min_dist, dist_new)

    return centers

    # # 初始化 min_dist：所有点到第一个中心的距离
    # if metric == "euclidean":
    #     min_dist = torch.norm(features - features[init_idx], dim=1)
    # elif metric == "cosine":
    #     min_dist = 1 - torch.matmul(features, features[init_idx])
    # else:
    #     raise ValueError
    #
    # for _ in range(1, k):
    #     idx = torch.argmax(min_dist).item()
    #     centers.append(idx)
    #
    #     if metric == "euclidean":
    #         dist_new = torch.norm(features - features[idx], dim=1)
    #     else:
    #         dist_new = 1 - torch.matmul(features, features[idx])
    #
    #     min_dist = torch.minimum(min_dist, dist_new)
    #
    # return centers

def k_center_greedy_with_one_init_quality(
    features: torch.Tensor,
    quality_scores: torch.Tensor,
    k: int,
    init_idx,
    metric: str = "euclidean",
    use_norm: bool = True,
    lambda_quality: float = 0.5,
    eps: float = 1e-12,
):
    """
    features: [N, D] torch tensor
    quality_scores: [N] torch tensor, larger is better
                    e.g. confidence / margin / negative entropy
    k: number of centers
    init_idx: 指定第一个中心点索引（None 表示随机）
    lambda_quality: 联合打分中质量项权重
                    joint = lambda_quality * quality + (1-lambda_quality) * distance
    """
    device = features.device
    N = features.size(0)

    assert features.ndim == 2, f"features must be [N, D], got {features.shape}"
    assert quality_scores.ndim == 1 and quality_scores.size(0) == N, \
        f"quality_scores must be [N], got {quality_scores.shape}"
    assert 0.0 <= lambda_quality <= 1.0, "lambda_quality must be in [0, 1]"
    assert k <= N, f"k={k} cannot exceed N={N}"

    quality_scores = quality_scores.to(device).float()

    if use_norm:
        features = F.normalize(features, p=2, dim=1)

    centers = []

    if init_idx is None:
        rand_idx = torch.randint(0, N, (1,), device=device).item()
        centers = [rand_idx]
    elif isinstance(init_idx, int):
        centers = [init_idx]
    else:
        if isinstance(init_idx, torch.Tensor):
            centers = init_idx.detach().cpu().tolist()
        else:
            centers = list(init_idx)

    if len(centers) > k:
        print(f"Warning: init_idx length ({len(centers)}) is larger than k ({k}). Returning init_idx.")
        return centers[:k]

    def compute_dist(center_idx):
        if metric == "euclidean":
            return torch.norm(features - features[center_idx], dim=1)
        elif metric == "cosine":
            return 1 - torch.matmul(features, features[center_idx])
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def minmax_normalize(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + eps)

    quality_norm = minmax_normalize(quality_scores)

    min_dist = torch.full((N,), float('inf'), device=device)

    for center_idx in centers:
        dist_new = compute_dist(center_idx)
        min_dist = torch.minimum(min_dist, dist_new)

    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    selected_mask[centers] = True

    num_needed = k - len(centers)

    for _ in range(num_needed):
        dist_norm = minmax_normalize(min_dist)
        joint_score = lambda_quality * quality_norm + (1.0 - lambda_quality) * dist_norm
        joint_score[selected_mask] = -1e9

        idx = torch.argmax(joint_score).item()
        centers.append(idx)
        selected_mask[idx] = True

        dist_new = compute_dist(idx)
        min_dist = torch.minimum(min_dist, dist_new)

    return centers

def k_center_greedy_with_init_quality_dynamic_lambda(
    features: torch.Tensor,
    quality_scores: torch.Tensor,
    k: int,
    init_idx=None,
    init_feature: torch.Tensor = None,
    metric: str = "euclidean",
    use_norm: bool = True,
    lambda_min: float = 0.2,
    lambda_max: float = 0.5,
    lambda_mode: str = "linear",   # "linear" or "quadratic"
    eps: float = 1e-12,
):
    """
    Quality-aware K-center with dynamic lambda.

    If init_idx is used:
        returns k selected indices (including init_idx)
    If init_feature is used:
        returns k-1 selected indices (virtual anchor does not count)

    Args:
        features: [N, D]
        quality_scores: [N], larger is better
        k: target number of centers
        init_idx: real sample index / indices
        init_feature: virtual anchor feature
        metric: "euclidean" or "cosine"
        use_norm: whether to normalize features / init_feature
        lambda_min: initial quality weight
        lambda_max: final quality weight
        lambda_mode: "linear" or "quadratic"
    """
    device = features.device
    N = features.size(0)

    assert features.ndim == 2, f"features must be [N, D], got {features.shape}"
    assert quality_scores.ndim == 1 and quality_scores.size(0) == N, \
        f"quality_scores must be [N], got {quality_scores.shape}"
    assert 0.0 <= lambda_min <= 1.0
    assert 0.0 <= lambda_max <= 1.0
    assert lambda_min <= lambda_max
    assert not (init_idx is None and init_feature is None), \
        "At least one of init_idx or init_feature must be provided."

    features = features.to(device).float()
    quality_scores = quality_scores.to(device).float()

    if use_norm:
        features = F.normalize(features, p=2, dim=1)

    def minmax_normalize(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + eps)

    def compute_dist_from_index(center_idx: int):
        center_vec = features[center_idx]
        if metric == "euclidean":
            return torch.norm(features - center_vec, dim=1)
        elif metric == "cosine":
            return 1.0 - torch.matmul(features, center_vec)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_dist_from_feature(center_feature: torch.Tensor):
        center_feature = center_feature.view(-1)
        if metric == "euclidean":
            return torch.norm(features - center_feature, dim=1)
        elif metric == "cosine":
            return 1.0 - torch.matmul(features, center_feature)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_lambda(step: int, total_steps: int):
        if total_steps <= 1:
            return lambda_max
        progress = step / (total_steps - 1)
        if lambda_mode == "linear":
            factor = progress
        elif lambda_mode == "quadratic":
            factor = progress ** 2
        else:
            raise ValueError(f"Unknown lambda_mode: {lambda_mode}")
        return lambda_min + (lambda_max - lambda_min) * factor

    quality_norm = minmax_normalize(quality_scores)

    centers = []
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    min_dist = torch.full((N,), float("inf"), device=device)

    if init_idx is not None:
        if isinstance(init_idx, int):
            centers = [init_idx]
        elif isinstance(init_idx, torch.Tensor):
            if init_idx.ndim == 0:
                centers = [int(init_idx.item())]
            else:
                centers = [int(x) for x in init_idx.detach().cpu().tolist()]
        else:
            centers = [int(x) for x in init_idx]

        if len(centers) > k:
            return centers[:k]

        for c in centers:
            dist_new = compute_dist_from_index(c)
            min_dist = torch.minimum(min_dist, dist_new)

        selected_mask[centers] = True
        # num_needed = k
        num_needed = k - len(centers)

    elif init_feature is not None:
        init_feature = init_feature.to(device).float().view(-1)
        if use_norm:
            init_feature = F.normalize(init_feature.unsqueeze(0), p=2, dim=1).squeeze(0)

        min_dist = compute_dist_from_feature(init_feature)

        # virtual anchor does not count as a real selected center
        num_needed = k - 1

    else:
        N1, D1 = features.shape

        # 随机选择第一个中心点
        start_idx = torch.randint(0, N1, (1,)).item()
        centers = [start_idx]

        if len(centers) > k:
            return centers[:k]

        for c in centers:
            dist_new = compute_dist_from_index(c)
            min_dist = torch.minimum(min_dist, dist_new)

        selected_mask[centers] = True
        num_needed = k - len(centers)


    assert num_needed >= 0

    for step in range(num_needed):
        dist_norm = minmax_normalize(min_dist)
        lambda_quality = get_lambda(step, num_needed)

        joint_score = lambda_quality * quality_norm + (1.0 - lambda_quality) * dist_norm
        joint_score[selected_mask] = -1e9

        idx = torch.argmax(joint_score).item()
        centers.append(idx)
        selected_mask[idx] = True

        dist_new = compute_dist_from_index(idx)
        min_dist = torch.minimum(min_dist, dist_new)

    return centers





def k_center_greedy_with_anchor(
        features: torch.Tensor,
        k: int,
        anchor_feature: torch.Tensor,
        metric: str = "euclidean",
        use_norm: bool = True
):
    """
    features: [N, D] torch tensor
    k: number of centers (including the anchor)
    anchor_feature: [D] or [1, D] torch tensor, 作为初始中心点
    """
    device = features.device
    N = features.size(0)

    # 1. 确保 anchor_feature 形状正确并移至设备
    anchor_feature = anchor_feature.to(device).view(-1)

    if use_norm:
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        anchor_feature = torch.nn.functional.normalize(anchor_feature, p=2, dim=0)

    # 辅助函数：计算所有点到指定 center (索引或特征向量) 的距离
    def compute_dist(center):
        # 如果 center 是 Tensor 向量，直接用；如果是索引，从 features 取
        if isinstance(center, int):
            center_vec = features[center]
        else:
            center_vec = center

        if metric == "euclidean":
            return torch.norm(features - center_vec, dim=1)
        elif metric == "cosine":
            return 1 - torch.matmul(features, center_vec)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # 2. 初始化：以 anchor_feature 为第一个中心
    # 计算所有点到 anchor 的距离，作为初始 min_dist
    min_dist = compute_dist(anchor_feature)

    # 记录已选中心 (这里以None表示它是虚拟锚点，或者你可以将其加入列表)
    # 如果你想把 anchor 也作为一个索引返回，需要额外逻辑去匹配 features 中最接近的点
    selected_indices = []

    # 3. 循环选择剩余的 K-1 个中心
    for _ in range(k - 1):
        # 找到距离当前中心集合（包括 anchor）最远的点
        idx = torch.argmax(min_dist).item()
        selected_indices.append(idx)

        # 更新 min_dist: 选取新的点后，更新最小距离
        dist_new = compute_dist(idx)
        min_dist = torch.minimum(min_dist, dist_new)

    return selected_indices




def herding(features, k):
    """
    Herding coreset selection (GPU Optimized)
    features: [N, D] torch tensor (already on GPU)
    k: number of samples to select
    return: selected indices (list)
    """
    device = features.device
    N, D = features.size()

    # 1. 计算全局均值，自动继承 device
    mean_vec = features.mean(dim=0)

    selected = []
    # 2. 初始化累积向量，明确指定 device
    cumulative = torch.zeros(D, device=device)

    # 预先分配一个 mask 以避免重复选择同一个样本
    # 也可以在循环内通过赋值极值实现
    mask = torch.ones(N, device=device, dtype=torch.bool)

    for i in range(k):
        # 当前已选数量（包含即将选入的一个）
        current_count = i + 1

        # 3. 计算每个样本加入后的平均值与全局均值的欧氏距离
        # 利用广播机制：(cumulative[D] + features[N, D]) -> [N, D]
        # / current_count -> [N, D]
        # - mean_vec[D] -> [N, D]
        candidate_means = (cumulative + features) / current_count
        dist = torch.norm(candidate_means - mean_vec, dim=1)

        # 4. 排除已选样本：将已选点的距离设为无穷大
        dist[~mask] = float('inf')

        # 5. 找到最能减少差距的样本
        idx = torch.argmin(dist).item()

        selected.append(idx)
        cumulative += features[idx]
        mask[idx] = False

    return selected


from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        封装数据以支持实时数据增强。
        images: numpy array 或 tensor
        labels: numpy array 或 tensor
        transform: torchvision.transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        # 1. 确保数据是 Tensor 类型 (假设输入是 float [0,1])
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()

        # 2. 应用增强 (Flip, Crop, Normalize)
        # 注意：transforms 期望输入是 (C, H, W) 格式
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)



class SyntheticDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy array or torch tensor, shape [N, C, H, W], value in [0,1]
        labels: numpy array or torch tensor, shape [N]
        """
        if isinstance(images, torch.Tensor):
            self.images = images.float()
        else:
            self.images = torch.from_numpy(images).float()

        if isinstance(labels, torch.Tensor):
            self.labels = labels.long()
        else:
            self.labels = torch.from_numpy(labels).long()

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]   # [C, H, W], already in [0,1]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y





class Evaluator(object):
    def __init__(self, config):

        # Set the computing device. Use GPU if available, otherwise use CPU. 只使用一个gpu进行evaluator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store the path for the statistics of sensitive data.
        self.sensitive_stats_path = config.sensitive_data.fid_stats

        # List of models used for evaluation.
        self.acc_models = ["resnet", "wrn", "resnext", "convnet"]

        # Save the configuration.
        self.config = config

        # Save the local rank for distributed settings.
        self.local_rank = config.setup.local_rank

        # Clear the GPU cache to free memory.
        torch.cuda.empty_cache()

    def train_classifier(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_val_loader,
             sensitive_test_loader, acc_models, semantic_public_dataset, classifier_save_path):

        self.acc_models = acc_models

        # Proceed only if this is the main process and a test loader is provided.
        if self.config.setup.global_rank != 0 or sensitive_test_loader is None:
            return

        # Check if the synthetic images have the expected resolution.
        # 如果合成图像的分辨率与敏感数据分辨率不相符合，则采用插值方法将合成数据变换成敏感数据的分辨率
        if synthetic_images.shape[-1] != self.config.sensitive_data.resolution:
            synthetic_images = F.interpolate(torch.from_numpy(synthetic_images),
                                             size=[self.config.sensitive_data.resolution,
                                                   self.config.sensitive_data.resolution]).numpy()

        # If the images are in color but only one channel is expected, convert to grayscale.
        # 如果合成图像是3通道的，但希望合成的是单通道的图
        if synthetic_images.shape[1] == 3 and self.config.sensitive_data.num_channels == 1:
            synthetic_images = 0.299 * synthetic_images[:, 2:, ...] + 0.587 * synthetic_images[:, 1:2, ...] + 0.114 * \
                               synthetic_images[:, :1, ...]

        acc_list = []

        # # If the image resolution is greater than 32, use only the first two models.
        # if synthetic_images.shape[-1] > 32:
        #     self.acc_models = self.acc_models[:2]

        # Loop over each model and compute accuracy metrics.
        for model_name in self.acc_models:
            set_global_seed(2333)
            if semantic_public_dataset is None:
                acc, test_acc_on_val, test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val = self.train_acc(
                model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader, classifier_save_path, semantic_public_dataset)
            else:
                acc, test_acc_on_val, test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val = self.train_acc_public(
                    model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader,
                    classifier_save_path, semantic_public_dataset)



            # Log the accuracy results.
            if sensitive_val_loader is not None:
                logging.info(
                    "The best acc of synthetic images on sensitive val and the corresponding acc on test dataset from {} is {} and {}".format(
                        model_name, acc, test_acc_on_val))
                logging.info(
                    "The best acc of synthetic images on noisy sensitive val and the corresponding acc on test dataset from {} is {} and {}".format(
                        model_name, best_noisy_acc, best_test_acc_on_noisy_val))
                acc_list.append(best_noisy_acc)

            else:
                logging.info(
                    "The best acc of synthetic images on val (synthetic images) and the corresponding acc on test dataset from {} is {} and {}".format(
                        model_name, acc, test_acc_on_val))
                acc_list.append(test_acc_on_val)
            logging.info("The best acc test dataset from {} is {}".format(model_name, test_acc_on_test))

        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        if sensitive_val_loader is not None:
            logging.info(
                f"The best acc of accuracy (adding noise to the results on the sensitive set of validation set) of synthetic images from resnet, wrn, and resnext are {acc_list}.")
        else:
            logging.info(
                f"The best acc of accuracy (using synthetic images as the validation set) of synthetic images from resnet, wrn, and resnext are {acc_list}.")

        logging.info(f"The average and std of accuracy of synthetic images are {acc_mean:.2f} and {acc_std:.2f}")

        torch.cuda.empty_cache()


    def eval_fidelity(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_val_loader,
                      sensitive_test_loader):

        # Proceed only if this is the main process and a test loader is provided.
        if str(self.device) != 'cuda:0' or sensitive_test_loader is None:
            return

        # Check if the synthetic images have the expected resolution.
        if synthetic_images.shape[-1] != self.config.sensitive_data.resolution:
            synthetic_images = F.interpolate(torch.from_numpy(synthetic_images),
                                             size=[self.config.sensitive_data.resolution,
                                                   self.config.sensitive_data.resolution]).numpy()

        # If the images are in color but only one channel is expected, convert to grayscale.
        if synthetic_images.shape[1] == 3 and self.config.sensitive_data.num_channels == 1:
            synthetic_images = 0.299 * synthetic_images[:, 2:, ...] + 0.587 * synthetic_images[:, 1:2, ...] + 0.114 * \
                               synthetic_images[:, :1, ...]

        fid, is_mean, fld, p, r, ir = self.visual_metric(synthetic_images, synthetic_labels, sensitive_train_loader,
                                                         sensitive_test_loader)
        logging.info("The FID of synthetic images is {}".format(fid))
        logging.info("The Inception Score of synthetic images is {}".format(is_mean))
        logging.info("The Precision and Recall of synthetic images is {} and {}".format(p, r))
        logging.info("The FLD of synthetic images is {}".format(fld))
        logging.info("The ImageReward of synthetic images is {}".format(ir))

        return fid, is_mean, p, r, fld, ir

    def visual_metric(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_test_loader):

        # Create an inception-based feature extractor. The save_path is based on the dataset name and resolution.
        feature_extractor = InceptionFeatureExtractor(
            save_path="dataset/{}_{}/".format(self.config.sensitive_data.name, self.config.sensitive_data.resolution))
        fc_layer = fid_inception_v3().fc
        fc_layer.eval()

        gen_images = torch.from_numpy(synthetic_images)

        # If the images have one channel, replicate it to create three channels.
        if gen_images.shape[1] == 1:
            gen_images = gen_images.repeat(1, 3, 1, 1)

        # Extract features from the synthetic images using the feature extractor.
        gen_feat = feature_extractor.get_tensor_features(gen_images)

        # Compute logits from the features without tracking gradients.
        with torch.no_grad():
            gen_logit = fc_layer(gen_feat).detach().cpu().numpy()
            gen_logit = np.exp(gen_logit) / np.sum(np.exp(gen_logit), 1, keepdims=True)

        # Try to obtain train and test features with a placeholder tensor.
        try:
            train_feat = feature_extractor.get_tensor_features(torch.tensor([0]), name="train")
            test_feat = feature_extractor.get_tensor_features(torch.tensor([0]), name="test")
        except:
            # If the placeholder method fails, load images from the sensitive loaders.
            train_images = []
            test_images = []
            for x, _ in sensitive_train_loader:
                train_images.append(x.float())
            for x, _ in sensitive_test_loader:
                test_images.append(x.float())

            # Concatenate the loaded images into tensors.
            train_images = torch.cat(train_images)
            test_images = torch.cat(test_images)

            # If the images have one channel, replicate it to get three channels.
            if train_images.shape[1] == 1:
                train_images = train_images.repeat(1, 3, 1, 1)
                test_images = test_images.repeat(1, 3, 1, 1)

            # Extract features from the train and test images.
            train_feat = feature_extractor.get_tensor_features(train_images, name="train")
            test_feat = feature_extractor.get_tensor_features(test_images, name="test")

        # Determine the number of unique classes in the synthetic labels.
        num_classes = len(set(synthetic_labels))

        # Load the image reward model.
        rm_model = RM.load("ImageReward-v1.0")
        ir = 0
        prompt = get_prompt(self.config.sensitive_data.name)

        # Compute the image reward score for each class without tracking gradients.
        with torch.no_grad():
            for cls in range(num_classes):

                # Select images for the current class, scale to 0-255, and convert to unsigned 8-bit integers.
                imgs = (synthetic_images[synthetic_labels == cls] * 255.).astype('uint8')
                imgs = np.transpose(imgs, (0, 2, 3, 1))
                if imgs.shape[-1] == 1:
                    imgs = np.repeat(imgs, 3, axis=-1)
                imgs = [Image.fromarray(img) for img in imgs]
                score = rm_model.score(prompt[cls], imgs)
                ir += np.sum(score)

        ir /= len(synthetic_images)

        is_mean, _ = compute_inception_score_from_logits(gen_logit)
        fid = FID().compute_metric(train_feat, None, gen_feat)
        fld = FLD(eval_feat="train").compute_metric(train_feat, test_feat, gen_feat)
        p = PrecisionRecall(mode="Precision", num_neighbors=4).compute_metric(train_feat, None,
                                                                              gen_feat)  # Default precision
        r = PrecisionRecall(mode="Recall", num_neighbors=4).compute_metric(train_feat, None, gen_feat)

        return fid, is_mean, fld, p, r, ir

    def pr_varying_k(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_val_loader,
                     sensitive_test_loader):
        # Proceed only if this is the main process and a test loader is provided.
        if str(self.device) != 'cuda:0' or sensitive_test_loader is None:
            return

        # Check if the synthetic images have the expected resolution.
        if synthetic_images.shape[-1] != self.config.sensitive_data.resolution:
            synthetic_images = F.interpolate(torch.from_numpy(synthetic_images),
                                             size=[self.config.sensitive_data.resolution,
                                                   self.config.sensitive_data.resolution]).numpy()

        # If the images are in color but only one channel is expected, convert to grayscale.
        if synthetic_images.shape[1] == 3 and self.config.sensitive_data.num_channels == 1:
            synthetic_images = 0.299 * synthetic_images[:, 2:, ...] + 0.587 * synthetic_images[:, 1:2, ...] + 0.114 * \
                               synthetic_images[:, :1, ...]

        # Create an inception-based feature extractor. The save_path is based on the dataset name and resolution.
        feature_extractor = InceptionFeatureExtractor(
            save_path="dataset/{}_{}/".format(self.config.sensitive_data.name, self.config.sensitive_data.resolution))

        gen_images = torch.from_numpy(synthetic_images)

        # If the images have one channel, replicate it to create three channels.
        if gen_images.shape[1] == 1:
            gen_images = gen_images.repeat(1, 3, 1, 1)

        # Extract features from the synthetic images using the feature extractor.
        gen_feat = feature_extractor.get_tensor_features(gen_images)

        # Try to obtain train and test features with a placeholder tensor.
        try:
            train_feat = feature_extractor.get_tensor_features(torch.tensor([0]), name="train")
        except:
            # If the placeholder method fails, load images from the sensitive loaders.
            train_images = []
            for x, _ in sensitive_train_loader:
                train_images.append(x.float())

            # Concatenate the loaded images into tensors.
            train_images = torch.cat(train_images)

            # If the images have one channel, replicate it to get three channels.
            if train_images.shape[1] == 1:
                train_images = train_images.repeat(1, 3, 1, 1)

            # Extract features from the train and test images.
            train_feat = feature_extractor.get_tensor_features(train_images, name="train")

        for k in [2, 7]:
            p = PrecisionRecall(mode="Precision", num_neighbors=k).compute_metric(train_feat, None,
                                                                                  gen_feat)  # Default precision
            r = PrecisionRecall(mode="Recall", num_neighbors=k).compute_metric(train_feat, None, gen_feat)
            logging.info("The Precision and Recall of synthetic images is {} and {} at k = {}".format(p, r, k))




    # cal_acc代码在少量合成数据下要结合mixup操作
    def train_acc_public(self, model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader,
                classifier_save_path, semantic_public_dataset=None):

        set_global_seed(2333)
        if semantic_public_dataset is not None:

            if 'cifar' in self.config.sensitive_data.name or 'celeba' in self.config.sensitive_data.name:
                public_dataset_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ])


            elif 'fmnist' in self.config.sensitive_data.name:
                trans = [
                    transforms.Resize(self.config.public_data.resolution),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
                trans = [transforms.Grayscale(num_output_channels=1)] + trans
                public_dataset_transform = transforms.Compose(trans)

            public_dataset = PublicDistillationDataset(
                selected_public_distillations=semantic_public_dataset,
                transform=public_dataset_transform
            )
            if 'cifar' in self.config.sensitive_data.name:
                batch_size = 128
            elif 'celeba' in self.config.sensitive_data.name:
                batch_size = 128
            else:
                batch_size = 256

            # 构建单卡场景下的dataset_loader
            public_dataset_loader = DataLoader(
                public_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

        # Set a fixed random seed and shuffle the synthetic images and labels.
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(len(synthetic_images))
        synthetic_images = synthetic_images[indices]
        synthetic_labels = synthetic_labels[indices]

        if sensitive_val_loader is not None:
            synthetic_images_train = synthetic_images
            synthetic_labels_train = synthetic_labels

        else:
            # Split the shuffled data into a training set and a validation set.
            synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
            synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]
            print("use syn val")

        num_classes = len(set(synthetic_labels))

        criterion = nn.CrossEntropyLoss()

        if 'cifar' in self.config.sensitive_data.name:
            batch_size = 128
            max_epoch = 200
            n_splits = 1
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            elif model_name == "resnet_18":
                model = resnet_18.ResNet18(num_classes=num_classes,
                                           channel=3, norm_type="batch")

            elif model_name == "dense121":
                model = dense121.DenseNet121(num_classes=num_classes, channel=3, norm_type="batch")


            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
        else:
            batch_size = 256
            if synthetic_images.shape[-1] == 64:
                n_splits = 16
            elif synthetic_images.shape[-1] == 128:
                n_splits = 64
            else:
                n_splits = 1
            max_epoch = 50
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes)
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                num_classes=num_classes, dropRate=0.3)
            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))
            if 'celeba' in self.config.sensitive_data.name:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

            if 'fmnist' in self.config.sensitive_data.name:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

        device = self.device
        model = model.to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)


        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_train).float(),
                                                torch.from_numpy(synthetic_labels_train).long()), shuffle=True,
                                  batch_size=batch_size // n_splits)

        if sensitive_val_loader is None:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(),
                                                  torch.from_numpy(synthetic_labels_val).long()), shuffle=False,
                                    batch_size=batch_size // n_splits)
            sensitive_val = False

            print("construct syn val")
        else:
            val_loader = sensitive_val_loader
            sensitive_val = True
            val_acc_list = []

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

        grad_accu_step = 0

        current_dir = Path(__file__).resolve().parent

        project_root = current_dir.parent

        classifier_save_path = project_root / classifier_save_path.lstrip('./')

        save_dir = str(classifier_save_path)

        save_path = os.path.join(save_dir, f"{model_name}_semantic_public_dataset_pretrain_val_test_epsilon_3.pt")
        #
        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            val_loss = 0
            total = 0
            correct = 0
            for _, (inputs, targets) in enumerate(public_dataset_loader):
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets) / n_splits
                train_loss += loss.item()
                loss.backward()
                grad_accu_step += 1

                if grad_accu_step == n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total * batch_size

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total * batch_size

            test_total = 0
            test_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = test_correct / test_total * 100

            logging.info(
                "Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch,
                                                                                                     train_acc,
                                                                                                     val_acc,
                                                                                                     test_acc,
                                                                                                     train_loss,
                                                                                                     val_loss))

            # If a sensitive validation loader is used, add Laplace noise for differential privacy.
            if sensitive_val:
                noisy_val_correct = val_correct + np.random.laplace(loc=0,
                                                                    scale=1 / (self.config.train.dp.epsilon / 3))
                if noisy_val_correct >= best_noisy_correct:
                    best_noisy_correct = noisy_val_correct
                    best_noisy_acc = noisy_val_correct / val_total * 100
                    best_test_acc_on_noisy_val = test_acc
                    logging.info(
                        "Epoch: {} Train acc: {} Noisy Val acc: {} Test acc{}".format(epoch,
                                                                                      train_acc,
                                                                                      best_noisy_acc,
                                                                                      best_test_acc_on_noisy_val
                                                                                      ))
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Successfully save model in {save_path}")
            else:
                best_noisy_acc = 0
                best_test_acc_on_noisy_val = 0

            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc

            # Restore the original model parameters from EMA.
            ema.restore(model.parameters())

        save_path = os.path.join(save_dir, f"{model_name}_semantic_public_dataset_pretrain_val_test_final.pt")
        torch.save(model.state_dict(), save_path)
        logging.info(f"Successfully save model in {save_path}")

        save_path = os.path.join(save_dir, f"{model_name}_semantic_public_dataset_pretrain_val_test_epsilon_3.pt")

        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint)

        set_global_seed(2333)

        # # # 2. 冻结所有特征提取层 (Backbone)
        # for param in model.layers.parameters():
        #     param.requires_grad = False
        #
        #
        # # 3. 确保分类头是开启梯度更新的
        # for param in model.classifier.parameters():
        #     param.requires_grad = True
        #
        # last_idx = model.depth - 1
        # # 解冻最后一层卷积
        # for param in model.layers['conv'][last_idx].parameters():
        #     param.requires_grad = True
        #
        # # 如果有 Norm 层，也一起解冻
        # if len(model.layers['norm']) > 0:
        #     for param in model.layers['norm'][last_idx].parameters():
        #         param.requires_grad = True

        del ema

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9, weight_decay=5e-4)

        head_params = [p for p in model.classifier.parameters() if p.requires_grad]
        head_param_ids = {id(p) for p in head_params}

        backbone_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and id(p) not in head_param_ids
        ]

        if 'cifar' in self.config.sensitive_data.name:
            max_epoch = 100

            optimizer = torch.optim.SGD(
                [{"params": backbone_params, "lr": 1e-3},
                 {"params": head_params, "lr": 1e-2}],
                momentum=0.9, weight_decay=5e-4, nesterov=True
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epoch,
                eta_min=1e-5
            )
        elif 'celeba' in self.config.sensitive_data.name or 'fmnist' in self.config.sensitive_data.name:
            max_epoch = 50
            if self.config.train.dp.epsilon == 1.0:
                optimizer = torch.optim.Adam(
                    [{"params": backbone_params, "lr": 1e-5},
                     {"params": head_params, "lr": 1e-5}],

                )
            else:
                optimizer = torch.optim.Adam(
                    [{"params": backbone_params, "lr": 1e-4},
                     {"params": head_params, "lr": 1e-5}],

                )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epoch,
                eta_min=1e-6
            )

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

        grad_accu_step = 0

        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            total = 0
            correct = 0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets) / n_splits
                train_loss += loss.item()
                loss.backward()
                grad_accu_step += 1

                if grad_accu_step == n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total * batch_size

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total * batch_size

            test_total = 0
            test_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = test_correct / test_total * 100

            logging.info(
                "Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch,
                                                                                                     train_acc,
                                                                                                     val_acc,
                                                                                                     test_acc,
                                                                                                     train_loss,
                                                                                                     val_loss))

            # If a sensitive validation loader is used, add Laplace noise for differential privacy.
            if sensitive_val:
                noisy_val_correct = val_correct + np.random.laplace(loc=0,
                                                                    scale=1 / (self.config.train.dp.epsilon / 3))
                if noisy_val_correct >= best_noisy_correct:
                    best_noisy_correct = noisy_val_correct
                    best_noisy_acc = noisy_val_correct / val_total * 100
                    best_test_acc_on_noisy_val = test_acc
                    logging.info(
                        "Epoch: {} Train acc: {} Noisy Val acc: {} Test acc{}".format(epoch,
                                                                                      train_acc,
                                                                                      best_noisy_acc,
                                                                                      best_test_acc_on_noisy_val
                                                                                      ))

                    save_path = os.path.join(save_dir,
                                             f"{model_name}_semantic_public_dataset_pretrain_val_test_epsilon_3_finetune_val_test_epsilon_3.pt")
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Successfully save model in {save_path}")


            else:
                best_noisy_acc = 0
                best_test_acc_on_noisy_val = 0

            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc

            # Restore the original model parameters from EMA.
            ema.restore(model.parameters())


        return best_acc, best_test_acc_on_val, best_test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val


    def train_acc(self, model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader, classifier_save_path, semantic_public_dataset = None):

        set_global_seed(2333)
        if semantic_public_dataset is not None:

            if 'cifar' in self.config.sensitive_data.name or 'celeba' in self.config.sensitive_data.name:
                public_dataset_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ])


            elif 'fmnist' in self.config.sensitive_data.name:
                trans = [
                    transforms.Resize(self.config.public_data.resolution),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
                trans = [transforms.Grayscale(num_output_channels=1)] + trans
                public_dataset_transform = transforms.Compose(trans)


            public_dataset = PublicDistillationDataset(
                selected_public_distillations=semantic_public_dataset,
                transform=public_dataset_transform
            )
            if 'cifar' in self.config.sensitive_data.name:
                batch_size = 128
            elif 'celeba' in self.config.sensitive_data.name:
                batch_size = 128
            else:
                batch_size = 256


            # 构建单卡场景下的dataset_loader
            public_dataset_loader = DataLoader(
                public_dataset,
                batch_size= batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

        # Set a fixed random seed and shuffle the synthetic images and labels.
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(len(synthetic_images))
        synthetic_images = synthetic_images[indices]
        synthetic_labels = synthetic_labels[indices]

        if sensitive_val_loader is not None:
            synthetic_images_train = synthetic_images
            synthetic_labels_train = synthetic_labels

        else:
            # Split the shuffled data into a training set and a validation set.
            synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
            synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]
            print("use syn val")

        num_classes = len(set(synthetic_labels))

        criterion = nn.CrossEntropyLoss()

        if 'cifar' in self.config.sensitive_data.name:
            batch_size = 128
            max_epoch = 200
            n_splits = 1
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            elif model_name == "resnet_18":
                model = resnet_18.ResNet18(num_classes=num_classes,
                                           channel=3, norm_type="batch")

            elif model_name == "dense121":
                model = dense121.DenseNet121(num_classes=num_classes, channel=3, norm_type="batch")


            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
        else:
            batch_size = 256
            if synthetic_images.shape[-1] == 64:
                n_splits = 16
            elif synthetic_images.shape[-1] == 128:
                n_splits = 64
            else:
                n_splits = 1
            max_epoch = 50
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes)
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                num_classes=num_classes, dropRate=0.3)
            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))
            if 'celeba' in self.config.sensitive_data.name:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

            if 'fmnist' in self.config.sensitive_data.name:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

        device = self.device
        model = model.to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_train).float(),
                                                torch.from_numpy(synthetic_labels_train).long()), shuffle=True,
                                  batch_size=batch_size // n_splits)

        if sensitive_val_loader is None:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(),
                                                  torch.from_numpy(synthetic_labels_val).long()), shuffle=False,
                                    batch_size=batch_size // n_splits)
            sensitive_val = False

            print("construct syn val")
        else:
            val_loader = sensitive_val_loader
            sensitive_val = True
            val_acc_list = []

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

        grad_accu_step = 0

        current_dir = Path(__file__).resolve().parent

        project_root = current_dir.parent

        classifier_save_path = project_root / classifier_save_path.lstrip('./')

        save_dir = str(classifier_save_path)

        save_path = os.path.join(save_dir, f"{model_name}_val_test_epsilon_3.pt")
        #
        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            val_loss = 0
            total = 0
            correct = 0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets) / n_splits
                train_loss += loss.item()
                loss.backward()
                grad_accu_step += 1

                if grad_accu_step == n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total * batch_size

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total * batch_size

            test_total = 0
            test_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = test_correct / test_total * 100

            logging.info(
                "Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch, train_acc,
                                                                                                     val_acc, test_acc,
                                                                                                     train_loss,
                                                                                                     val_loss))

            # If a sensitive validation loader is used, add Laplace noise for differential privacy.
            if sensitive_val:
                noisy_val_correct = val_correct + np.random.laplace(loc=0, scale=1 / (self.config.train.dp.epsilon / 3))
                if noisy_val_correct >= best_noisy_correct:
                    best_noisy_correct = noisy_val_correct
                    best_noisy_acc = noisy_val_correct / val_total * 100
                    best_test_acc_on_noisy_val = test_acc
                    logging.info(
                        "Epoch: {} Train acc: {} Noisy Val acc: {} Test acc{}".format(epoch,
                                                                                      train_acc,
                                                                                      best_noisy_acc,
                                                                                      best_test_acc_on_noisy_val
                                                                                      ))
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Successfully save model in {save_path}")
            else:
                best_noisy_acc = 0
                best_test_acc_on_noisy_val = 0

            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc

            # Restore the original model parameters from EMA.
            ema.restore(model.parameters())

        save_path = os.path.join(save_dir, f"{model_name}_val_test_final.pt")
        torch.save(model.state_dict(), save_path)
        logging.info(f"Successfully save model in {save_path}")


        return best_acc, best_test_acc_on_val, best_test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val

    # 在不同ipc下计算
    def eval_dataset_distillation(self, synthetic_images, synthetic_labels, sensitive_train_loader,
                                  sensitive_val_loader,
                                  sensitive_test_loader, acc_models, classifier_save_path,
                                  random_ipc=True, k_center_ipc=False, k_means_ipc = False, classifier_guidance = True ,soft_label=True):

        # Proceed only if this is the main process and a test loader is provided.
        if self.config.setup.global_rank != 0 or sensitive_test_loader is None:
            return

        # Check if the synthetic images have the expected resolution.
        # 如果合成图像的分辨率与敏感数据分辨率不相符合，则采用插值方法将合成数据变换成敏感数据的分辨率
        if synthetic_images.shape[-1] != self.config.sensitive_data.resolution:
            synthetic_images = F.interpolate(torch.from_numpy(synthetic_images),
                                             size=[self.config.sensitive_data.resolution,
                                                   self.config.sensitive_data.resolution]).numpy()

        # If the images are in color but only one channel is expected, convert to grayscale.
        # 如果合成图像是3通道的，但希望合成的是单通道的图
        if synthetic_images.shape[1] == 3 and self.config.sensitive_data.num_channels == 1:
            synthetic_images = 0.299 * synthetic_images[:, 2:, ...] + 0.587 * synthetic_images[:, 1:2, ...] + 0.114 * \
                               synthetic_images[:, :1, ...]

        acc_list = []

        # If the image resolution is greater than 32, use only the first two models.
        if synthetic_images.shape[-1] > 32:
            self.acc_models = self.acc_models[:2]

        self.acc_models = acc_models
        # self.acc_models = ["resnet", "wrn", "resnext", "convnet"]
        ipc_list = [1, 10, 50]
        # ipc_list = [50, 100]
        # ipc_list = []
        print(ipc_list)

        num_classes = len(set(synthetic_labels))
        index_list = []

        for i in range(num_classes):
            index_list.append([0])

        test_acc_total_list = []
        anchor_images = []
        anchor_features = []
        tmp_ipc = True


        for ipc in ipc_list:

            # Loop over each model and compute accuracy metrics.
            for model_name in self.acc_models:
                acc_list_inner = []
                test_acc_on_val_list = []
                test_acc_on_test_list = []
                best_noisy_acc_list = []
                best_test_acc_on_noisy_val_list = []
                test_acc_list = []

                # 测试阶段repeat=1
                tmp_index_list = copy.deepcopy(index_list)
                if ipc == 1:
                    repeat_times = 3
                else:
                    repeat_times = 3

                for repeat_time in range(repeat_times):
                    set_global_seed(2333 + repeat_time)
                    if random_ipc == False:
                        if tmp_ipc == True:

                            acc, test_acc_on_val, test_acc_on_test, _, test_acc, anchor_images= self.cal_acc_with_dataset_distillation_under_core_set(
                                model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader,
                                ipc, tmp_index_list, anchor_images, classifier_save_path, k_center=k_center_ipc, k_means=k_means_ipc, is_classifier_guidance = classifier_guidance, is_soft_label = soft_label)
                            tmp_ipc = False
                        else:
                            acc, test_acc_on_val, test_acc_on_test, _ , test_acc, _ = self.cal_acc_with_dataset_distillation_under_core_set(
                                model_name, synthetic_images, synthetic_labels, sensitive_val_loader,sensitive_test_loader,
                                ipc, tmp_index_list,anchor_images, classifier_save_path, k_center=k_center_ipc, k_means=k_means_ipc,
                                is_classifier_guidance=classifier_guidance, is_soft_label=soft_label)


                        print(index_list)
                    else:
                        acc, test_acc_on_val, test_acc_on_test, test_acc = self.cal_acc_with_dataset_distillation(
                            model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader,
                            ipc, classifier_save_path,
                            is_soft_label = soft_label)
                    acc_list_inner.append(acc)
                    test_acc_on_val_list.append(test_acc_on_val)
                    test_acc_on_test_list.append(test_acc_on_test)
                    test_acc_list.append(test_acc)


                acc = np.array(acc_list_inner).mean()
                test_acc_on_val = np.array(test_acc_on_val_list).mean()
                test_acc_on_test = np.array(test_acc_on_test_list).mean()

                test_acc = np.array(test_acc_list).mean()
                test_acc_total_list.append(test_acc)
                test_acc_std = np.array(test_acc_list).std()
                test_acc_total_list.append(test_acc_std)

                logging.info("The test_acc_total_list: {}".format(test_acc_total_list))

            torch.cuda.empty_cache()

    # 计算不同ipc下
    def cal_acc_with_dataset_distillation(self, model_name, synthetic_images, synthetic_labels,
                                          sensitive_val_loader, sensitive_test_loader, ipc, classifier_save_path, is_soft_label=True):

        # Set a fixed random seed and shuffle the synthetic images and labels.
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(len(synthetic_images))
        synthetic_images = synthetic_images[indices]
        synthetic_labels = synthetic_labels[indices]

        if sensitive_val_loader is not None:
            synthetic_images_train = synthetic_images
            synthetic_labels_train = synthetic_labels
        else:
            # Split the shuffled data into a training set and a validation set.
            synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
            synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]

        num_classes = len(set(synthetic_labels))


        synthetic_images_train_ipc = []
        synthetic_labels_train_ipc = []

        if ipc is not None:

            synthetic_images_train_torch = torch.from_numpy(synthetic_images_train)
            synthetic_labels_train_torch = torch.from_numpy(synthetic_labels_train)

            for cls in range(num_classes):
                # 找到该类所有的 index（按照原顺序）
                cls_indices = (synthetic_labels_train_torch == cls).nonzero(as_tuple=True)[0]

                # 取前 ipc 个；如果不够，则全部保留
                selected = cls_indices[:ipc]

                synthetic_images_train_ipc.append(synthetic_images_train_torch[selected])
                synthetic_labels_train_ipc.append(synthetic_labels_train_torch[selected])

            # 拼接成最终结果
            synthetic_images_train_ipc = torch.cat(synthetic_images_train_ipc, dim=0).numpy()
            synthetic_labels_train_ipc = torch.cat(synthetic_labels_train_ipc, dim=0).numpy()

            synthetic_images_train = synthetic_images_train_ipc
            synthetic_labels_train = synthetic_labels_train_ipc

        logging.info("IPC 数据大小:{}, {}".format(synthetic_images_train.shape, synthetic_labels_train.shape))
        print("IPC 数据大小:", synthetic_images_train.shape, synthetic_labels_train.shape)

        criterion = nn.CrossEntropyLoss()

        if ipc is None:
            batch_size = 128
        elif ipc >= 50:
            batch_size = 100
        elif ipc >= 10:
            batch_size = 50
        elif ipc > 0:
            batch_size = 15

        if num_classes == 10:
            batch_size *= 1
        if num_classes == 100:
            batch_size *= 2

        if 'cifar' in self.config.sensitive_data.name:

            if self.config.sensitive_data.name == "cifar10":
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2023, 0.1994, 0.2010)

            if is_soft_label == True:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, padding=4),  # 填充后随机裁剪，增加位置不变性
                    # transforms.Normalize(mean, std)
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),  # 填充后随机裁剪，增加位置不变性
                    # transforms.Normalize(mean, std)
                ])
            # normalizer = transforms.Normalize(mean, std)

            if ipc is None:
                batch_size = 128
                max_epoch = 200
            else:
                print("开始mixup训练")
                max_epoch = 500

            n_splits = 1
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)


            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')


            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)


            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            current_dir = Path(__file__).resolve().parent

            project_root = current_dir.parent

            classifier_save_path = project_root / classifier_save_path.lstrip('./')

            teacher_dir = str(classifier_save_path)



            teacher_model = CN.ConvNet(num_classes,
                                       net_norm="instance",
                                       net_depth=3,
                                       net_width=128,
                                       channel=synthetic_images.shape[1],
                                       im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            teacher_path = os.path.join(teacher_dir, f"{model_name}_val_test_epsilon.pt")
            state_dict = torch.load(teacher_path, map_location=self.device)
            teacher_model.load_state_dict(state_dict)

            teacher_model.eval()

            # freeze all layers
            for param in teacher_model.parameters():
                param.requires_grad = False

            logging.info(f"Successfully test load model from {teacher_path}")

            if ipc is None:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
            else:
                optimizer = torch.optim.AdamW(
                    get_parameters(model),
                    lr=0.001,
                    betas=[0.9, 0.999],
                    weight_decay=0.01,
                )
                scheduler = LambdaLR(
                    optimizer,
                    lambda step: 0.5 * (1.0 + math.cos(math.pi * step / max_epoch / 2))
                    if step <= max_epoch
                    else 0,
                    last_epoch=-1,
                )

        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),  # 填充后随机裁剪，增加位置不变性
                # transforms.Normalize(mean, std)
            ])

            if ipc is None:
                batch_size = 256
                max_epoch = 50
            else:
                print("开始cutmix训练")

                max_epoch = 500
                if 'mnist' in self.setup.sensitive_data.name:
                    max_epoch = 1000

            # batch_size = 256
            if synthetic_images.shape[-1] == 64:
                n_splits = 16
            elif synthetic_images.shape[-1] == 128:
                n_splits = 64
            else:
                n_splits = 1

            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)


            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')


            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)


            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            current_dir = Path(__file__).resolve().parent

            project_root = current_dir.parent

            classifier_save_path = project_root / classifier_save_path.lstrip('./')

            teacher_dir = str(classifier_save_path)


            teacher_model = CN.ConvNet(num_classes,
                                       net_norm="instance",
                                       net_depth=3,
                                       net_width=128,
                                       channel=synthetic_images.shape[1],
                                       im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))
            teacher_path = os.path.join(teacher_dir, f"{model_name}_val_test_epsilon.pt")


            # teacher_path = os.path.join(teacher_dir, f"{model_name}.pt")
            state_dict = torch.load(teacher_path, map_location=self.device)
            teacher_model.load_state_dict(state_dict)
            teacher_model = teacher_model.to(self.device)

            teacher_model.eval()

            # freeze all layers
            for param in teacher_model.parameters():
                param.requires_grad = False

            logging.info(f"Successfully test load model from {teacher_path}")

            if ipc is None:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
            else:

                optimizer = torch.optim.AdamW(
                    get_parameters(model),
                    lr=0.001,
                    betas=[0.9, 0.999],
                    weight_decay=0.01,
                )
                scheduler = LambdaLR(
                    optimizer,
                    lambda step: 0.5 * (1.0 + math.cos(math.pi * step / max_epoch / 2))
                    if step <= max_epoch
                    else 0,
                    last_epoch=-1,
                )

        device = torch.device("cuda:0")
        model = model.to(device)
        teacher_model = teacher_model.to(device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        train_dataset = AugmentedDataset(synthetic_images_train, synthetic_labels_train, transform=train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size // n_splits)


        if sensitive_val_loader is None:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(),
                                                  torch.from_numpy(synthetic_labels_val).long()), shuffle=True,
                                    batch_size=batch_size // n_splits)
            sensitive_val = False
        else:
            val_loader = sensitive_val_loader
            sensitive_val = True
            val_acc_list = []

        loss_function_kl = nn.KLDivLoss(reduction="batchmean")

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

        grad_accu_step = 0

        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            total = 0
            correct = 0

            for _, (inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                if ipc is None:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                else:
                    # cutmix setting

                    beta = 1.0
                    lam = np.random.beta(beta, beta)
                    rand_index = random_indices(targets, nclass=num_classes)
                    target_a = targets
                    target_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs_new = inputs.clone()
                    inputs_new[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                    outputs = model(inputs_new)

                    if is_soft_label == True:
                        soft_mix_label = teacher_model(inputs_new)
                        soft_mix_label = F.softmax(soft_mix_label / 20, dim=1)

                        soft_pred_mix_label = F.log_softmax(outputs / 20, dim=1)

                        loss = loss_function_kl(soft_pred_mix_label, soft_mix_label) / n_splits

                    else:

                        loss = (criterion(outputs, target_a) / n_splits) * ratio + (
                                    criterion(outputs, target_b) / n_splits) * (1. - ratio)


                train_loss += loss.item()
                loss.backward()
                grad_accu_step += 1

                if grad_accu_step == n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total * batch_size

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    # inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # inputs = normalizer(inputs)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total * batch_size

            test_total = 0
            test_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    # inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # inputs = normalizer(inputs)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = test_correct / test_total * 100

            logging.info("Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              test_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))


            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc

            # Restore the original model parameters from EMA.
            ema.restore(model.parameters())

        return best_acc, best_test_acc_on_val, best_test_acc_on_test, test_acc



    # 在不同IPC下计算K-centers结果
    def cal_acc_with_dataset_distillation_under_core_set(self, model_name, synthetic_images, synthetic_labels,
                                                         sensitive_val_loader, sensitive_test_loader, ipc,
                                                         index_list, anchor_images, classifier_save_path, k_center, k_means, is_classifier_guidance=True,
                                                         is_soft_label=True):

        # Set a fixed random seed and shuffle the synthetic images and labels.
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(len(synthetic_images))
        synthetic_images = synthetic_images[indices]
        synthetic_labels = synthetic_labels[indices]

        if sensitive_val_loader is not None:
            synthetic_images_train = synthetic_images
            synthetic_labels_train = synthetic_labels
        else:
            # Split the shuffled data into a training set and a validation set.
            synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
            synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]

        num_classes = len(set(synthetic_labels))


        if ipc is None:
            batch_size = 128
        elif ipc >= 50:
            batch_size = 100
        elif ipc >= 10:
            batch_size = 50
        elif ipc > 0:
            batch_size = 15

        if num_classes == 10:
            batch_size *= 1
        if num_classes == 100:
            batch_size *= 2

        if 'cifar' in self.config.sensitive_data.name:

            if self.config.sensitive_data.name == "cifar10":
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2023, 0.1994, 0.2010)

            if is_soft_label == True:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, padding=4),  # 填充后随机裁剪，增加位置不变性
                    # transforms.Normalize(mean, std)
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),  # 填充后随机裁剪，增加位置不变性
                    # transforms.Normalize(mean, std)
                ])
            # normalizer = transforms.Normalize(mean, std)

            if ipc is None:
                batch_size = 128
                max_epoch = 200
            else:
                print("开始cutmix训练")
                max_epoch = 500

            n_splits = 1
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)



            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')



            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)


            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))


            elif model_name == "resnet_18":
                model = resnet_18.ResNet18(num_classes= num_classes,
                                            channel=3, norm_type="batch")

            elif model_name == "dense121":
                model = dense121.DenseNet121(num_classes=num_classes,channel=3,norm_type="batch",)



            current_dir = Path(__file__).resolve().parent

            project_root = current_dir.parent

            classifier_save_path = project_root / classifier_save_path.lstrip('./')

            teacher_dir = str(classifier_save_path)



            public_assist_teacher_model = CN.ConvNet(num_classes,
                                                     net_norm="instance",
                                                     net_depth=3,
                                                     net_width=128,
                                                     channel=synthetic_images.shape[1],
                                                     im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            # public_assist_teacher_model = resnet_18.ResNet18(num_classes= num_classes,
            #                                 channel=3, norm_type="batch")

            # public_assist_teacher_model = dense121.DenseNet121(num_classes=num_classes, channel=3, norm_type="batch")



            public_assist_teacher_path = os.path.join(teacher_dir,
                                                      f"{model_name}_semantic_public_dataset_pretrain_val_test_epsilon_3_finetune_val_test_epsilon_3.pt")


            state_dict = torch.load(public_assist_teacher_path, map_location=self.device)
            public_assist_teacher_model.load_state_dict(state_dict)
            public_assist_teacher_model = public_assist_teacher_model.to(self.device)

            public_assist_teacher_model.eval()

            # freeze all layers
            for param in public_assist_teacher_model.parameters():
                param.requires_grad = False



            teacher_model = CN.ConvNet(num_classes,
                                       net_norm="instance",
                                       net_depth=3,
                                       net_width=128,
                                       channel=synthetic_images.shape[1],
                                       im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))
            # teacher_model = resnet_18.ResNet18(num_classes=num_classes,
            #                                                  channel=3, norm_type="batch")

            # teacher_model = dense121.DenseNet121(num_classes=num_classes, channel=3, norm_type="batch")



            teacher_path = os.path.join(teacher_dir, f"{model_name}_val_test_epsilon_3.pt")

            # teacher_path = os.path.join(teacher_dir, f"{model_name}.pt")
            state_dict = torch.load(teacher_path, map_location=self.device)
            teacher_model.load_state_dict(state_dict)
            teacher_model = teacher_model.to(self.device)

            teacher_model.eval()

            # freeze all layers
            for param in teacher_model.parameters():
                param.requires_grad = False

            logging.info(f"Successfully test load model from {teacher_path}")

            if ipc is None:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
            else:
                optimizer = torch.optim.AdamW(
                    get_parameters(model),
                    lr=0.001,
                    betas=[0.9, 0.999],
                    weight_decay=0.01,
                )
                scheduler = LambdaLR(
                    optimizer,
                    lambda step: 0.5 * (1.0 + math.cos(math.pi * step / max_epoch / 2))
                    if step <= max_epoch
                    else 0,
                    last_epoch=-1,
                )

        else:

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),  # 填充后随机裁剪，增加位置不变性
                # transforms.Normalize(mean, std)
            ])

            if ipc is None:
                batch_size = 256
                max_epoch = 50
            else:
                print("开始cutmix训练")
                max_epoch = 500
                if 'mnist' in self.config.sensitive_data.name:
                    max_epoch = 1000


            # batch_size = 256
            if synthetic_images.shape[-1] == 64:
                n_splits = 16
            elif synthetic_images.shape[-1] == 128:
                n_splits = 64
            else:
                n_splits = 1

            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)


            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')


            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)


            elif model_name == "convnet-3":
                model = CN.ConvNet(num_classes,
                                   net_norm="instance",
                                   net_depth=3,
                                   net_width=128,
                                   channel=synthetic_images.shape[1],
                                   im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))

            current_dir = Path(__file__).resolve().parent

            project_root = current_dir.parent

            classifier_save_path = project_root / classifier_save_path.lstrip('./')

            teacher_dir = str(classifier_save_path)


            public_assist_teacher_model = CN.ConvNet(num_classes,
                                                     net_norm="instance",
                                                     net_depth=3,
                                                     net_width=128,
                                                     channel=synthetic_images.shape[1],
                                                     im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))
            public_assist_teacher_path = os.path.join(teacher_dir,
                                                      f"{model_name}_semantic_public_dataset_pretrain_val_test_epsilon_3_finetune_val_test_epsilon_3.pt")

            # teacher_path = os.path.join(teacher_dir, f"{model_name}.pt")
            state_dict = torch.load(public_assist_teacher_path, map_location=self.device)
            public_assist_teacher_model.load_state_dict(state_dict)
            public_assist_teacher_model = public_assist_teacher_model.to(self.device)

            public_assist_teacher_model.eval()

            # freeze all layers
            for param in public_assist_teacher_model.parameters():
                param.requires_grad = False

            teacher_model = CN.ConvNet(num_classes,
                                       net_norm="instance",
                                       net_depth=3,
                                       net_width=128,
                                       channel=synthetic_images.shape[1],
                                       im_size=(synthetic_images.shape[2], synthetic_images.shape[2]))
            teacher_path = os.path.join(teacher_dir,
                                        f"{model_name}_val_test_epsilon_3.pt")

            # teacher_path = os.path.join(teacher_dir, f"{model_name}.pt")
            state_dict = torch.load(teacher_path, map_location=self.device)
            teacher_model.load_state_dict(state_dict)
            teacher_model = teacher_model.to(self.device)

            teacher_model.eval()

            # freeze all layers
            for param in teacher_model.parameters():
                param.requires_grad = False

            logging.info(f"Successfully test load model from {teacher_path}")

            if ipc is None:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
            else:

                optimizer = torch.optim.AdamW(
                    get_parameters(model),
                    lr=0.001,
                    betas=[0.9, 0.999],
                    weight_decay=0.01,
                )
                scheduler = LambdaLR(
                    optimizer,
                    lambda step: 0.5 * (1.0 + math.cos(math.pi * step / max_epoch / 2))
                    if step <= max_epoch
                    else 0,
                    last_epoch=-1,
                )



        synthetic_images_train_ipc = []
        synthetic_labels_train_ipc = []

        if ipc is not None:
            if is_classifier_guidance == False:
                feature_extractor = InceptionFeatureExtractor()

                synthetic_images_train_torch = torch.from_numpy(synthetic_images_train).float()
                synthetic_labels_train_torch = torch.from_numpy(synthetic_labels_train)

                for cls in range(num_classes):

                    cls_mask = synthetic_labels_train_torch == cls
                    cls_images = synthetic_images_train_torch[cls_mask].to(self.device)
                    print(cls_images.shape)

                    if cls_images.size(0) <= ipc:
                        selected = torch.arange(cls_images.size(0))
                    else:
                        with torch.no_grad():
                            # cls_images = cls_images * 2. - 1.
                            # _, feats = feature_extractor_model(cls_images, return_features=True)
                            # feats = feature_extractor.get_tensor_features(cls_images)

                            probs, feats, pred = extract_logits_features_batched(
                                model=teacher_model,
                                images_cpu=cls_images,
                                device=self.device,
                            )


                            # feats = feats.cpu()
                        if k_center == True:
                            selected_idx = k_center_greedy(feats, ipc)
                        elif k_means == True:
                            selected_idx = kmeans_sampling(feats, ipc)
                        else:
                            selected_idx = herding(feats, ipc)
                        selected = torch.tensor(selected_idx, device=self.device)

                    synthetic_images_train_ipc.append(cls_images[selected].cpu())
                    synthetic_labels_train_ipc.append(
                        torch.full((selected.size(0),), cls, dtype=torch.long))

                # 拼接成最终结果
                synthetic_images_train_ipc = torch.cat(synthetic_images_train_ipc, dim=0).numpy()
                synthetic_labels_train_ipc = torch.cat(synthetic_labels_train_ipc, dim=0).numpy()

                synthetic_images_train = synthetic_images_train_ipc
                synthetic_labels_train = synthetic_labels_train_ipc

            else:
                synthetic_images_train_torch = torch.from_numpy(synthetic_images_train).float()
                synthetic_labels_train_torch = torch.from_numpy(synthetic_labels_train)

                feature_extractor = InceptionFeatureExtractor()

                for cls in range(num_classes):
                    cls_mask = synthetic_labels_train_torch == cls
                    cls_images_cpu = synthetic_images_train_torch[cls_mask].cpu()
                    print(cls_images_cpu.shape)

                    if cls_images_cpu.size(0) <= ipc:
                        selected_idx = torch.arange(cls_images_cpu.size(0))

                    else:
                        with torch.no_grad():

                            if ipc == 1:

                                current_model = public_assist_teacher_model

                            else:
                                current_model = teacher_model


                            probs, features, pred = extract_logits_features_batched(
                                model=current_model,
                                images_cpu=cls_images_cpu,
                                device=self.device,
                            )

                            correct_mask = pred == cls

                            if correct_mask.sum() == 0:
                                # 极端情况下：teacher 对该类全部错误
                                # 直接用该类的logit最大的样本
                                cls_score = probs[:, cls]
                                selected_idx = torch.topk(cls_score, ipc).indices
                            else:
                                # 只在 pred==cls 的样本上计算 margin
                                probs_correct = probs[correct_mask]  # [M, C]
                                features_correct = features[correct_mask]

                                # 记录下真实的indices
                                correct_indices = torch.nonzero(correct_mask, as_tuple=False).squeeze(1)

                                # 目标类概率 p1（此时 top1 就是 cls）
                                p1 = probs_correct[:, cls]  # [M]

                                # 次大概率 p2：除 cls 外的最大概率
                                probs_others = probs_correct.clone()
                                probs_others[:, cls] = -1.0  # 排除 cls（softmax 概率 >=0，因此置 -1 安全）
                                p2 = probs_others.max(dim=1).values  # [M]


                                cls_score = p1
                                # cls_score = p1 - p2  # [M]
                                # entropy = -torch.sum(probs_correct * torch.log(probs_correct + 1e-10), dim=1)
                                #
                                # cls_score = -entropy




                                if cls_score.size(0) <= ipc:
                                    selected_idx = correct_indices
                                else:
                                    if ipc <= 0:
                                        topk_idx = torch.topk(cls_score, ipc).indices

                                        selected_idx = correct_indices[topk_idx]
                                    else:
                                        # top1_idx = torch.topk(cls_score, 1).indices
                                        #
                                        # selected_1_idx = correct_indices[top1_idx]

                                        # 1. 按置信度从高到低排序
                                        _, sorted_idx = torch.sort(cls_score, descending=True)

                                        # 2. 取“高到中间（前 ipc * 5）”作为候选域
                                        M = sorted_idx.size(0)
                                        if ipc <= 100:
                                            candidate_end = min(ipc * 5, M)

                                        candidate_indices = sorted_idx[:candidate_end]

                                        # 3. 准备 K-center 的输入
                                        # 候选域的特征
                                        candidate_features = features_correct[candidate_indices]
                                        candidate_quality_scores = cls_score[candidate_indices]
                                        # 候选域在原图 cls_images 中的真实索引
                                        candidate_real_indices = correct_indices[candidate_indices]

                                        # 4. 确定锚点（候选域中 Margin 最大的点）
                                        # 因为 sorted_idx 已经按 Margin 降序，所以 candidate_features 的第 0 个就是锚点
                                        anchor_idx_in_candidate = index_list[cls]

                                        # 5. 执行 K-center (已经在候选域内筛选，所以 k = ipc)
                                        # 返回的是在 candidate_features 中的局部索引
                                        actual_k = min(ipc, candidate_features.size(0))

                                        if ipc == 1:
                                            selected_local_indices = k_center_greedy_with_init_quality_dynamic_lambda(
                                                features=candidate_features,
                                                quality_scores=candidate_quality_scores,
                                                k=actual_k,
                                                init_idx=anchor_idx_in_candidate,
                                                metric="cosine",
                                                use_norm=True,
                                                lambda_min=0,
                                                lambda_max=0,
                                                lambda_mode="linear"
                                            )
                                        else:
                                            _, anchor_feature = teacher_model(
                                                anchor_images[cls].to(self.device) * 2. - 1., return_features=True)
                                            anchor_feature = anchor_feature.squeeze(0)
                                            selected_local_indices = k_center_greedy_with_init_quality_dynamic_lambda(
                                                features=candidate_features,
                                                quality_scores=candidate_quality_scores,
                                                k=actual_k,
                                                init_feature=anchor_feature,
                                                # init_idx=anchor_idx_in_candidate,
                                                metric="cosine",
                                                use_norm=True,
                                                lambda_min=0,
                                                lambda_max=0,
                                                lambda_mode="linear"
                                            )


                                        # selected_local_indices = k_center_greedy(candidate_features, actual_k, use_norm=True)

                                        # selected_local_indices = herding(candidate_features, actual_k)
                                        #
                                        # selected_local_indices = kmeans_sampling(candidate_features, actual_k)

                                        index_list[cls] = selected_local_indices
                                        # 6. 映射回原始 cls_images 的全局索引
                                        # selected_idx = [candidate_real_indices[i] for i in selected_local_indices[1:]]
                                        selected_idx = candidate_real_indices[selected_local_indices]


                    if ipc == 1:
                        anchor_images.append(cls_images_cpu[selected_idx].cpu())

                        synthetic_images_train_ipc.append(cls_images_cpu[selected_idx].cpu())
                        synthetic_labels_train_ipc.append(
                                            torch.full((selected_idx.size(0),), cls, dtype=torch.long)
                                        )

                    else:
                        synthetic_images_train_ipc.append(anchor_images[cls])
                        synthetic_images_train_ipc.append(cls_images_cpu[selected_idx].cpu())
                        synthetic_labels_train_ipc.append(
                                            torch.full((selected_idx.size(0) + 1,), cls, dtype=torch.long)
                                        )
                        # synthetic_labels_train_ipc.append(
                        #     torch.full((selected_idx.size(0),), cls, dtype=torch.long)
                        # )


                    # synthetic_images_train_ipc.append(cls_images_cpu[selected_idx].cpu())
                    # synthetic_labels_train_ipc.append(
                    #     torch.full((selected_idx.size(0),), cls, dtype=torch.long)
                    # )
                    #
                    # del cls_images_cpu
                    # if 'probs' in locals():
                    #     del probs
                    # if 'features' in locals():
                    #     del features
                    # if 'pred' in locals():
                    #     del pred
                    # torch.cuda.empty_cache()

                # 拼接
                synthetic_images_train_ipc = torch.cat(synthetic_images_train_ipc, dim=0).numpy()
                synthetic_labels_train_ipc = torch.cat(synthetic_labels_train_ipc, dim=0).numpy()

                synthetic_images_train = synthetic_images_train_ipc
                synthetic_labels_train = synthetic_labels_train_ipc

        logging.info("IPC 数据大小:{}, {}".format(synthetic_images_train.shape, synthetic_labels_train.shape))
        print("IPC 数据大小:", synthetic_images_train.shape, synthetic_labels_train.shape)

        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda:0")
        model = model.to(device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        train_dataset = AugmentedDataset(synthetic_images_train, synthetic_labels_train, transform=train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size // n_splits)


        if sensitive_val_loader is None:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(),
                                                  torch.from_numpy(synthetic_labels_val).long()), shuffle=True,
                                    batch_size=batch_size // n_splits)
            sensitive_val = False
        else:
            val_loader = sensitive_val_loader
            sensitive_val = True
            val_acc_list = []

        loss_function_kl = nn.KLDivLoss(reduction="batchmean")

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

        grad_accu_step = 0

        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            total = 0
            correct = 0

            for _, (inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self.device) , targets.to(self.device)
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                if ipc is None:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                else:
                    # cutmix setting
                    beta = 1.0
                    lam = np.random.beta(beta, beta)
                    rand_index = random_indices(targets, nclass=num_classes)
                    target_a = targets
                    target_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs_new = inputs.clone()
                    inputs_new[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                    outputs = model(inputs_new)

                    if is_soft_label == True:

                        soft_mix_label = teacher_model(inputs_new)

                        soft_mix_label = F.softmax(soft_mix_label / 20, dim=1)

                        soft_pred_mix_label = F.log_softmax(outputs / 20, dim=1)

                        loss = loss_function_kl(soft_pred_mix_label, soft_mix_label) / n_splits

                    else:

                        loss = (criterion(outputs, target_a) / n_splits) * ratio + (
                                criterion(outputs, target_b) / n_splits) * (1. - ratio)


                train_loss += loss.item()
                loss.backward()
                grad_accu_step += 1

                if grad_accu_step == n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total * batch_size

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total * batch_size

            test_total = 0
            test_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = test_correct / test_total * 100

            logging.info("Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              test_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))


            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc

            # Restore the original model parameters from EMA.
            ema.restore(model.parameters())

        print(index_list)

        return best_acc, best_test_acc_on_val, best_test_acc_on_test, index_list, test_acc, anchor_images

    def eval(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_val_loader,
             sensitive_test_loader):

        # Proceed only if this is the main process and a test loader is provided.
        if self.config.setup.global_rank != 0 or sensitive_test_loader is None:
            return

        # Check if the synthetic images have the expected resolution.
        if synthetic_images.shape[-1] != self.config.sensitive_data.resolution:
            synthetic_images = F.interpolate(torch.from_numpy(synthetic_images),
                                             size=[self.config.sensitive_data.resolution,
                                                   self.config.sensitive_data.resolution]).numpy()

        # If the images are in color but only one channel is expected, convert to grayscale.
        if synthetic_images.shape[1] == 3 and self.config.sensitive_data.num_channels == 1:
            synthetic_images = 0.299 * synthetic_images[:, 2:, ...] + 0.587 * synthetic_images[:, 1:2, ...] + 0.114 * \
                               synthetic_images[:, :1, ...]

        acc_list = []

        # If the image resolution is greater than 32, use only the first two models.
        if synthetic_images.shape[-1] > 32:
            self.acc_models = self.acc_models[:2]

        # Loop over each model and compute accuracy metrics.
        for model_name in self.acc_models:
            acc, test_acc_on_val, test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val = self.cal_acc(
                model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader)

            # Log the accuracy results.
            if sensitive_val_loader is not None:
                logging.info(
                    "The best acc of synthetic images on sensitive val and the corresponding acc on test dataset from {} is {} and {}".format(
                        model_name, acc, test_acc_on_val))
                logging.info(
                    "The best acc of synthetic images on noisy sensitive val and the corresponding acc on test dataset from {} is {} and {}".format(
                        model_name, acc, test_acc_on_val))
            else:
                logging.info(
                    "The best acc of synthetic images on val (synthetic images) and the corresponding acc on test dataset from {} is {} and {}".format(
                        model_name, acc, test_acc_on_val))
            logging.info("The best acc test dataset from {} is {}".format(model_name, test_acc_on_test))
            acc_list.append(test_acc_on_val)

        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        if sensitive_val_loader is not None:
            logging.info(
                f"The best acc of accuracy (adding noise to the results on the sensitive set of validation set) of synthetic images from resnet, wrn, and resnext are {acc_list}.")
        else:
            logging.info(
                f"The best acc of accuracy (using synthetic images as the validation set) of synthetic images from resnet, wrn, and resnext are {acc_list}.")

        logging.info(f"The average and std of accuracy of synthetic images are {acc_mean:.2f} and {acc_std:.2f}")

        torch.cuda.empty_cache()

        # Calculate visual metrics: FID, Inception Score, FLD, precision, recall, and ImageReward.
        fid, is_mean, fld, p, r, ir = self.visual_metric(synthetic_images, synthetic_labels, sensitive_train_loader,
                                                         sensitive_test_loader)
        logging.info("The FID of synthetic images is {}".format(fid))
        logging.info("The Inception Score of synthetic images is {}".format(is_mean))
        logging.info("The Precision and Recall of synthetic images is {} and {}".format(p, r))
        logging.info("The FLD of synthetic images is {}".format(fld))
        logging.info("The ImageReward of synthetic images is {}".format(ir))

    def cal_acc(self, model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader):

        # Set a fixed random seed and shuffle the synthetic images and labels.
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(len(synthetic_images))
        synthetic_images = synthetic_images[indices]
        synthetic_labels = synthetic_labels[indices]

        # Split the shuffled data into a training set and a validation set.
        synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
        synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]

        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()

        if 'cifar' in self.config.sensitive_data.name:
            batch_size = 126
            max_epoch = 200
            n_splits = 1
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes, depth=164, block_name='BasicBlock')
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8,
                                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
        else:
            batch_size = 256
            if synthetic_images.shape[-1] == 64:
                n_splits = 16
            elif synthetic_images.shape[-1] == 128:
                n_splits = 64
            else:
                n_splits = 1
            max_epoch = 50
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                   num_classes=num_classes, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                               num_classes=num_classes)
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2],
                                num_classes=num_classes, dropRate=0.3)

            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

        device = torch.device("cuda:0")
        model = model.to(device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_train).float(),
                                                torch.from_numpy(synthetic_labels_train).long()), shuffle=True,
                                  batch_size=batch_size // n_splits)

        if sensitive_val_loader is None:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(),
                                                  torch.from_numpy(synthetic_labels_val).long()), shuffle=True,
                                    batch_size=batch_size // n_splits)
            sensitive_val = False
        else:
            val_loader = sensitive_val_loader
            sensitive_val = True
            val_acc_list = []

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

        grad_accu_step = 0

        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            total = 0
            correct = 0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)

                if grad_accu_step == 0:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets) / n_splits
                train_loss += loss.item()
                loss.backward()
                grad_accu_step += 1

                if grad_accu_step == n_splits:
                    optimizer.step()
                    grad_accu_step = 0
                    ema.update(model.parameters())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total * batch_size

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / n_splits
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total * batch_size

            test_total = 0
            test_correct = 0

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32)
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

            test_acc = test_correct / test_total * 100

            logging.info(
                "Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch, train_acc,
                                                                                                     val_acc, test_acc,
                                                                                                     train_loss,
                                                                                                     val_loss))

            # If a sensitive validation loader is used, add Laplace noise for differential privacy.
            if sensitive_val:
                noisy_val_correct = val_correct + np.random.laplace(loc=0, scale=1 / self.config.train.dp.epsilon)
                if noisy_val_correct >= best_noisy_correct:
                    best_noisy_correct = noisy_val_correct
                    best_noisy_acc = noisy_val_correct / val_total * 100
                    best_test_acc_on_noisy_val = test_acc
            else:
                best_noisy_acc = 0
                best_test_acc_on_noisy_val = 0

            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc

            # Restore the original model parameters from EMA.
            ema.restore(model.parameters())

        return best_acc, best_test_acc_on_val, best_test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val


    def cal_acc_no_dp(self, sensitive_train_loader, sensitive_test_loader):
        print(sensitive_train_loader, sensitive_test_loader)
        if sensitive_test_loader is None or sensitive_train_loader is None:
            return

        batch_size = 128
        lr = 1e-4
        max_epoch = 200

        train_loader = DataLoader(sensitive_train_loader.dataset, batch_size=batch_size, shuffle=True)
        test_loader = sensitive_test_loader

        num_classes = self.config.sensitive_data.n_classes
        c = self.config.sensitive_data.num_channels
        img_size = self.config.sensitive_data.resolution

        acc_list = []

        for model_name in self.acc_models:

            logging.info(f'model type:{model_name}')

            if 'cifar' in self.config.sensitive_data.name:

                batch_size = 256
                max_epoch = 200

                if model_name == "wrn":
                    model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes, depth=28, widen_factor=10,
                                       dropRate=0.3)
                elif model_name == "resnet":
                    model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes, depth=164,
                                   block_name='BasicBlock')
                elif model_name == "resnext":
                    model = ResNeXt(in_c=c, img_size=img_size, cardinality=8, depth=28, num_classes=num_classes,
                                    widen_factor=10, dropRate=0.3)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
            else:

                batch_size = 126
                max_epoch = 50

                if model_name == "wrn":
                    model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes, dropRate=0.3)
                elif model_name == "resnet":
                    model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes, block_name='BasicBlock')
                elif model_name == "resnext":
                    model = ResNeXt(in_c=c, img_size=img_size, num_classes=num_classes, dropRate=0.3)

                optimizer = optim.Adam(model.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

            # Move model to device (GPU/CPU)
            model = model.to(self.device)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()

            # optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Training the model
            best_acc = 0.0
            for epoch in range(max_epoch):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = inputs.float() * 2. - 1.
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    # Calculate statistics
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                scheduler.step()

                train_acc = 100. * correct / total
                logging.info(
                    f"Epoch [{epoch + 1}/{max_epoch}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

                # Testing the model
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        inputs = inputs.float() * 2. - 1.
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                test_acc = 100. * correct / total
                logging.info(f"Test Accuracy: {test_acc:.2f}%")

                # Return the best accuracy
                if test_acc > best_acc:
                    best_acc = test_acc

            logging.info("The best acc of original images from {} is {}".format(model_name, best_acc))

            acc_list.append(best_acc)

        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        logging.info(f"The best acc of accuracy of synthetic images from resnet, wrn, and resnext are {acc_list}.")
        logging.info(f"The average and std of accuracy of synthetic images are {acc_mean:.2f} and {acc_std:.2f}")


def compute_inception_score_from_logits(logits, splits=1):
    scores = []

    for i in range(splits):
        part = logits[(i * logits.shape[0] // splits):((i + 1) * logits.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        kl = np.mean([entropy(part[i, :], py) for i in range(part.shape[0])])
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


def get_prompt(data_name: str):
    if data_name.startswith("mnist"):
        return ["A grayscale image of a handwritten digit 0", "A grayscale image of a handwritten digit 1",
                "an image of hand-written 2", "an image of hand-written 3", "an image of hand-written 4",
                "an image of hand-written 5", "an image of hand-written 6", "an image of hand-written 7",
                "an image of hand-written 8", "an image of hand-written 9"]

    elif data_name.startswith("fmnist"):
        return ["A grayscale image of a T-shirt", "A grayscale image of a handwritten digit Trouser",
                "an image of hand-written Pullover", "an image of hand-written Dress", "an image of hand-written Coat",
                "an image of hand-written Sandal", "an image of hand-written Shirt", "an image of hand-written Sneaker",
                "an image of hand-written Bag", "an image of hand-written Ankle boot"]

    elif data_name.startswith("cifar100"):
        cifar100_y = '''Superclass	Classes
        aquatic mammals	beaver, dolphin, otter, seal, whale
        fish	aquarium fish, flatfish, ray, shark, trout
        flowers	orchids, poppies, roses, sunflowers, tulips
        food containers	bottles, bowls, cans, cups, plates
        fruit and vegetables	apples, mushrooms, oranges, pears, sweet peppers
        household electrical devices	clock, computer keyboard, lamp, telephone, television
        household furniture	bed, chair, couch, table, wardrobe
        insects	bee, beetle, butterfly, caterpillar, cockroach
        large carnivores	bear, leopard, lion, tiger, wolf
        large man-made outdoor things	bridge, castle, house, road, skyscraper
        large natural outdoor scenes	cloud, forest, mountain, plain, sea
        large omnivores and herbivores	camel, cattle, chimpanzee, elephant, kangaroo
        medium-sized mammals	fox, porcupine, possum, raccoon, skunk
        non-insect invertebrates	crab, lobster, snail, spider, worm
        people	baby, boy, girl, man, woman
        reptiles	crocodile, dinosaur, lizard, snake, turtle
        small mammals	hamster, mouse, rabbit, shrew, squirrel
        trees	maple, oak, palm, pine, willow
        vehicles 1	bicycle, bus, motorcycle, pickup truck, train
        vehicles 2	lawn-mower, rocket, streetcar, tank, tractor'''
        class_lins = cifar100_y.split("\n")

        prompt = []

        for line in class_lins[1:]:
            super_class, child_classes = line.split("\t")
            for child_class in child_classes.split(", "):
                prompt.append("An image of {}".format(child_class))
        return prompt

    elif data_name.startswith("cifar10"):
        return ["An image of an airplane", "An image of an automobile", "An image of a bird", "An image of a cat",
                "An image of a deer", "An image of a dog", "An image of a frog", "An image of a horse",
                "An image of a ship", "An image of a truck"]

    elif data_name.startswith("eurosat"):
        return ["A remote sensing image of an industrial area", "A remote sensing image of a residential area",
                "A remote sensing image of an annual crop area", "A remote sensing image of a permanent crop area",
                "A remote sensing image of a river area", "A remote sensing image of a sea or lake area",
                "A remote sensing image of a herbaceous veg. area", "A remote sensing image of a highway area",
                "A remote sensing image of a pasture area", "A remote sensing image of a forest area"]

    elif data_name.startswith("celeba_male"):
        return ["An image of a female face", "An image of a male face"]

    elif data_name.startswith("camelyon"):
        return ["A normal lymph node image", "A lymph node histopathology image"]

    else:
        raise NotImplementedError
