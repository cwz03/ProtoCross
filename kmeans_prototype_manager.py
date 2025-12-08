# kmeans_prototype_manager.py
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Optional, Tuple, Dict
import torch.nn.functional as F


class KMeansPrototypeManager:
    def __init__(self, num_subtypes: int, feature_dim: int, device: torch.device):
        self.num_subtypes = num_subtypes
        self.feature_dim = feature_dim
        self.device = device
        self.current_stage = 1

        # 初始化原型
        self.prototypes = {
            'mri': [None] * num_subtypes,
            'pet': [None] * num_subtypes
        }
        self.cn_prototypes = {
            'mri': None,
            'pet': None
        }

        # 存储所有样本的亚型标签映射
        self.subtype_mapping_mri = {}
        self.subtype_mapping_pet = {}

        # 存储样本索引到subject_id的映射
        self.index_to_subject = {}

        # K-means模型
        self.kmeans_mri = KMeans(n_clusters=num_subtypes, random_state=42, n_init=10)
        self.kmeans_pet = KMeans(n_clusters=num_subtypes, random_state=42, n_init=10)

        # 存储原始特征用于计算轮廓系数
        self.ad_mri_features_np = None
        self.ad_pet_features_np = None

        self.initialized = False

    def set_stage(self, stage: int):
        """设置当前训练阶段"""
        self.current_stage = stage

    def initialize_with_kmeans(self, mri_features: torch.Tensor, pet_features: torch.Tensor,
                               labels: torch.Tensor, subject_ids: List[str], ad_indices: List[int]):
        """使用K-means聚类初始化亚型原型和标签映射"""
        print("使用K-means聚类初始化亚型原型...")

        # 存储索引映射
        for idx, subject_id in enumerate(subject_ids):
            self.index_to_subject[idx] = subject_id

        # 提取AD样本的特征和subject_id
        ad_mri_features = mri_features[ad_indices]
        ad_pet_features = pet_features[ad_indices]
        ad_subject_ids = [subject_ids[i] for i in ad_indices]

        # 转换为numpy数组并保存用于轮廓系数计算
        self.ad_mri_features_np = ad_mri_features.numpy()
        self.ad_pet_features_np = ad_pet_features.numpy()

        print(f"AD样本数量: {len(ad_indices)}")
        print(f"MRI特征形状: {self.ad_mri_features_np.shape}")
        print(f"PET特征形状: {self.ad_pet_features_np.shape}")

        # 分别对MRI和PET特征进行K-means聚类
        print("对MRI特征进行K-means聚类...")
        subtype_labels_mri = self.kmeans_mri.fit_predict(self.ad_mri_features_np)

        print("对PET特征进行K-means聚类...")
        subtype_labels_pet = self.kmeans_pet.fit_predict(self.ad_pet_features_np)

        # 构建亚型标签映射
        for idx, subject_id in enumerate(ad_subject_ids):
            self.subtype_mapping_mri[subject_id] = subtype_labels_mri[idx]
            self.subtype_mapping_pet[subject_id] = subtype_labels_pet[idx]

        # 计算聚类中心作为原型
        for subtype in range(self.num_subtypes):
            # MRI原型
            mri_mask = (subtype_labels_mri == subtype)
            if mri_mask.sum() > 0:
                mri_prototype = torch.tensor(self.ad_mri_features_np[mri_mask].mean(axis=0),
                                             dtype=torch.float32).to(self.device)
                self.prototypes['mri'][subtype] = F.normalize(mri_prototype.unsqueeze(0), dim=1).squeeze(0)

            # PET原型
            pet_mask = (subtype_labels_pet == subtype)
            if pet_mask.sum() > 0:
                pet_prototype = torch.tensor(self.ad_pet_features_np[pet_mask].mean(axis=0),
                                             dtype=torch.float32).to(self.device)
                self.prototypes['pet'][subtype] = F.normalize(pet_prototype.unsqueeze(0), dim=1).squeeze(0)

        # 计算CN原型（所有CN样本的均值）
        cn_mask = (labels == 0)  # 假设CN是类别0
        if cn_mask.sum() > 0:
            cn_mri_features = mri_features[cn_mask]
            cn_pet_features = pet_features[cn_mask]

            self.cn_prototypes['mri'] = F.normalize(
                torch.tensor(cn_mri_features.mean(dim=0), dtype=torch.float32).unsqueeze(0).to(self.device),
                dim=1
            ).squeeze(0)
            self.cn_prototypes['pet'] = F.normalize(
                torch.tensor(cn_pet_features.mean(dim=0), dtype=torch.float32).unsqueeze(0).to(self.device),
                dim=1
            ).squeeze(0)

        # 打印聚类结果
        self._print_clustering_stats(subtype_labels_mri, subtype_labels_pet)

        self.initialized = True
        print("K-means亚型初始化完成!")

    def _print_clustering_stats(self, labels_mri: np.ndarray, labels_pet: np.ndarray):
        """打印聚类统计信息"""
        print("\n聚类统计信息:")
        print("MRI亚型分布:")
        unique_mri, counts_mri = np.unique(labels_mri, return_counts=True)
        for subtype, count in zip(unique_mri, counts_mri):
            print(f"  亚型 {subtype}: {count} 个样本 ({count / len(labels_mri) * 100:.1f}%)")

        print("PET亚型分布:")
        unique_pet, counts_pet = np.unique(labels_pet, return_counts=True)
        for subtype, count in zip(unique_pet, counts_pet):
            print(f"  亚型 {subtype}: {count} 个样本 ({count / len(labels_pet) * 100:.1f}%)")

        # 正确计算轮廓系数
        print("\n聚类质量评估:")
        if len(np.unique(labels_mri)) > 1 and len(labels_mri) > 1:
            try:
                sil_mri = silhouette_score(self.ad_mri_features_np, labels_mri)
                print(f"MRI轮廓系数: {sil_mri:.4f}")
            except Exception as e:
                print(f"MRI轮廓系数计算失败: {e}")
        else:
            print("MRI样本数量不足，无法计算轮廓系数")

        if len(np.unique(labels_pet)) > 1 and len(labels_pet) > 1:
            try:
                sil_pet = silhouette_score(self.ad_pet_features_np, labels_pet)
                print(f"PET轮廓系数: {sil_pet:.4f}")
            except Exception as e:
                print(f"PET轮廓系数计算失败: {e}")
        else:
            print("PET样本数量不足，无法计算轮廓系数")

        # 计算簇间距离
        self._print_cluster_distances()

    def _print_cluster_distances(self):
        """打印簇间距离信息"""
        print("\n簇间距离评估:")

        # MRI簇间距离
        if hasattr(self.kmeans_mri, 'cluster_centers_'):
            mri_centers = self.kmeans_mri.cluster_centers_
            print("MRI簇间距离:")
            for i in range(len(mri_centers)):
                for j in range(i + 1, len(mri_centers)):
                    dist = np.linalg.norm(mri_centers[i] - mri_centers[j])
                    print(f"  亚型 {i} 与 亚型 {j} 距离: {dist:.4f}")

        # PET簇间距离
        if hasattr(self.kmeans_pet, 'cluster_centers_'):
            pet_centers = self.kmeans_pet.cluster_centers_
            print("PET簇间距离:")
            for i in range(len(pet_centers)):
                for j in range(i + 1, len(pet_centers)):
                    dist = np.linalg.norm(pet_centers[i] - pet_centers[j])
                    print(f"  亚型 {i} 与 亚型 {j} 距离: {dist:.4f}")

    def get_batch_subtype_labels(self, subject_ids: List[str], labels: torch.Tensor, modality: str) -> torch.Tensor:
        """获取batch中样本的亚型标签"""
        batch_size = len(subject_ids)
        subtype_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        if modality == 'mri':
            mapping = self.subtype_mapping_mri
        else:
            mapping = self.subtype_mapping_pet

        for i, subject_id in enumerate(subject_ids):
            if labels[i] == 1 and subject_id in mapping:  # AD样本且有亚型标签
                subtype_labels[i] = mapping[subject_id]
            # CN样本保持为0（表示CN原型）

        return subtype_labels

    def get_prototypes_for_batch(self, labels: torch.Tensor, subtype_labels: torch.Tensor, modality: str):
        """获取batch样本对应的原型"""
        batch_size = labels.size(0)
        prototypes = []
        target_indices = []  # 每个样本对应的原型索引

        for i in range(batch_size):
            if labels[i] == 0:  # CN样本 -> CN原型
                prototype = self.cn_prototypes[modality]
                if prototype is not None:
                    prototypes.append(prototype)
                    target_indices.append(0)  # CN原型索引为0
                else:
                    # 如果没有CN原型，使用第一个亚型原型作为占位符
                    prototypes.append(self.prototypes[modality][0])
                    target_indices.append(0)
            else:  # AD样本 -> 对应亚型原型
                subtype = subtype_labels[i].item()
                prototype = self.prototypes[modality][subtype]
                if prototype is not None:
                    prototypes.append(prototype)
                    target_indices.append(subtype + 1)  # 亚型原型索引从1开始
                else:
                    # 如果该亚型原型不存在，使用第一个亚型原型
                    prototypes.append(self.prototypes[modality][0])
                    target_indices.append(1)

        if prototypes:
            prototypes_tensor = torch.stack(prototypes)
            target_indices_tensor = torch.tensor(target_indices, dtype=torch.long).to(self.device)
            return prototypes_tensor, target_indices_tensor
        else:
            return None, None

    def compute_prototype_similarities(self, features: torch.Tensor, labels: torch.Tensor,
                                       subtype_labels: torch.Tensor, modality: str) -> torch.Tensor:
        """计算特征与所有相关原型的相似度"""
        batch_size = features.size(0)

        # 收集所有相关原型（CN原型 + 所有亚型原型）
        all_prototypes = []

        # 添加CN原型
        if self.cn_prototypes[modality] is not None:
            all_prototypes.append(self.cn_prototypes[modality])

        # 添加所有亚型原型
        for subtype in range(self.num_subtypes):
            if self.prototypes[modality][subtype] is not None:
                all_prototypes.append(self.prototypes[modality][subtype])

        if not all_prototypes:
            return torch.zeros(batch_size, 1).to(self.device)

        prototypes_tensor = torch.stack(all_prototypes)  # [num_prototypes, feature_dim]

        # 计算所有样本与所有原型的相似度
        features_normalized = F.normalize(features, dim=1)  # [batch_size, feature_dim]
        prototypes_normalized = F.normalize(prototypes_tensor, dim=1)  # [num_prototypes, feature_dim]

        similarities = torch.mm(features_normalized, prototypes_normalized.t())  # [batch_size, num_prototypes]

        return similarities