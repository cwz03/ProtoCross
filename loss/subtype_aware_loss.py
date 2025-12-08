# subtype_aware_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubtypeAwareProtoLoss(nn.Module):
    """亚型感知原型对比损失 - 支持CN和AD样本"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, prototypes, subtype_labels, labels, cn_prototype=None):
        """
        Args:
            features: 样本特征 [batch_size, feature_dim]
            prototypes: 亚型原型列表 [num_subtypes, feature_dim]
            subtype_labels: 亚型标签 [batch_size]
            labels: 真实标签 [batch_size] (0: CN, 1: AD)
            cn_prototype: CN原型 [feature_dim]
        """
        batch_size = features.size(0)

        # 构建所有原型（CN原型 + 亚型原型）
        all_prototypes = []
        prototype_indices = []

        # 添加CN原型（如果存在）
        if cn_prototype is not None:
            all_prototypes.append(cn_prototype)
            prototype_indices.append(0)  # CN原型索引

        # 添加亚型原型
        num_subtypes = len(prototypes)
        for i in range(num_subtypes):
            if prototypes[i] is not None:
                all_prototypes.append(prototypes[i])
                prototype_indices.append(i + 1)  # 亚型原型索引从1开始

        if not all_prototypes:
            return torch.tensor(0.0, device=features.device)

        prototypes_tensor = torch.stack(all_prototypes)  # [num_total_prototypes, feature_dim]
        num_total_prototypes = prototypes_tensor.size(0)

        # 计算相似度矩阵
        features_normalized = F.normalize(features, dim=1)
        prototypes_normalized = F.normalize(prototypes_tensor, dim=1)

        similarity_matrix = torch.mm(features_normalized, prototypes_normalized.t()) / self.temperature
        # [batch_size, num_total_prototypes]

        # 构建目标标签
        target_labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)

        for i in range(batch_size):
            if labels[i] == 0:  # CN样本 -> 目标为CN原型
                target_labels[i] = 0  # CN原型索引
            else:  # AD样本 -> 目标为对应亚型原型
                subtype = subtype_labels[i].item()
                target_labels[i] = subtype + 1  # 亚型原型索引

        # 计算原型对比损失
        loss = self.criterion(similarity_matrix, target_labels)

        return loss


class SubtypeAwareMWCL(nn.Module):
    """亚型感知多模态加权对比损失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, mri_features, pet_features, labels, subtype_mri, subtype_pet, clinical_similarity):
        """
        完整的亚型感知多模态对比损失
        """
        batch_size = mri_features.size(0)

        # 构建亚型感知的权重矩阵
        weight_matrix = self._build_subtype_aware_weights(
            labels, subtype_mri, subtype_pet, clinical_similarity
        )

        # 计算四种模态组合的对比损失
        total_loss = 0.0
        modality_pairs = [
            (mri_features, pet_features, 'mri_pet'),
            (pet_features, mri_features, 'pet_mri'),
            (mri_features, mri_features, 'mri_mri'),
            (pet_features, pet_features, 'pet_pet')
        ]

        for features_i, features_j, pair_name in modality_pairs:
            loss_pair = self._compute_contrastive_loss(
                features_i, features_j, weight_matrix
            )
            total_loss += loss_pair

        return total_loss / len(modality_pairs)

    def _build_subtype_aware_weights(self, labels, subtype_mri, subtype_pet, clinical_similarity):
        """构建亚型感知的权重矩阵"""
        batch_size = labels.size(0)
        device = labels.device

        # 初始化权重矩阵
        weights = torch.zeros(batch_size, batch_size, device=device)

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue

                # 基础条件：同类样本
                if labels[i] == labels[j]:
                    if labels[i] == 0:  # 两个都是CN
                        weights[i, j] = clinical_similarity[i, j]
                    else:  # 两个都是AD
                        # 额外条件：相同亚型
                        if subtype_mri[i] == subtype_mri[j] and subtype_pet[i] == subtype_pet[j]:
                            weights[i, j] = clinical_similarity[i, j]
                        else:
                            weights[i, j] = 0.0
                else:
                    weights[i, j] = 0.0

        # 行归一化
        row_sums = weights.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        weights = weights / row_sums

        return weights

    def _compute_contrastive_loss(self, features_i, features_j, weight_matrix):
        """计算单个模态对的对比损失"""
        batch_size = features_i.size(0)

        # 计算相似度矩阵
        features_i_norm = F.normalize(features_i, dim=1)
        features_j_norm = F.normalize(features_j, dim=1)

        similarity_matrix = torch.mm(features_i_norm, features_j_norm.t()) / self.temperature

        # 计算log softmax
        log_softmax = F.log_softmax(similarity_matrix, dim=1)

        # 计算加权对比损失
        loss = -torch.sum(weight_matrix * log_softmax) / batch_size

        return loss