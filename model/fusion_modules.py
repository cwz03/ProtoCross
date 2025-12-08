import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=8)

    def forward(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(0)


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(CrossAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=8)

    def forward(self, x1, x2):
        # x1 Q
        # x2 K V
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)

        attn_output, _ = self.attention(query=x1, key=x2, value=x2)

        return attn_output.squeeze(0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # Hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits


class SumFusion(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=768, dim=768, output_dim=2, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=768, dim=768, output_dim=2, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


class CrossAttention(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(CrossAttention, self).__init__()
        self.cross_attention = CrossAttentionModule(input_dim)
        self.fc_out = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, y):
        mri_pet = self.cross_attention(x, y)
        pet_mri = self.cross_attention(y, x)
        output = torch.cat((mri_pet, pet_mri), dim=1)
        output = self.fc_out(output)
        return x, y, output

class ClinicalGuideCrossAttention(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(ClinicalGuideCrossAttention, self).__init__()
        self.cross_attention1 = CrossAttentionModule(input_dim)
        self.cross_attention2 = CrossAttentionModule(input_dim)
        self.clinical_encoder = nn.Linear(in_features=4, out_features=768)
        self.fc_out = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, y, z):
        z = self.clinical_encoder(z)
        mri = self.cross_attention1(z, x)
        pet = self.cross_attention2(z, y)
        output = torch.cat((mri, pet), dim=1)
        output = self.fc_out(output)
        return x, y, output


import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyFusion(nn.Module):
    """
    基于混合不确定性度量的动态融合模块
    结合熵和KL散度评估模态质量
    """

    def __init__(self, input_dim, output_dim, num_modalities=2, beta=0.5):
        super(UncertaintyFusion, self).__init__()
        self.num_modalities = num_modalities
        self.beta = beta  # 熵和KL散度的权重平衡参数
        self.epsilon = 1e-8

        # 最终的分类层
        self.classifier = nn.Linear(input_dim * num_modalities, output_dim)

    def compute_hybrid_uncertainty(self, logits_list):
        """
        计算混合不确定性度量
        Args:
            logits_list: 各模态的logits列表 [logits_mri, logits_pet]
        Returns:
            uncertainties: 各模态的不确定性值
            weights: 融合权重
        """
        batch_size = logits_list[0].size(0)
        uncertainties = []
        normalized_entropies = []
        normalized_kls = []

        # 计算各模态的概率分布
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]

        # 1. 计算熵
        entropies = []
        for probs in probs_list:
            entropy = -torch.sum(probs * torch.log(probs + self.epsilon), dim=1)
            entropies.append(entropy)

        # 归一化熵
        for entropy in entropies:
            mean_e = torch.mean(entropy)
            std_e = torch.std(entropy)
            normalized_e = (entropy - mean_e) / (std_e + self.epsilon)
            normalized_entropies.append(normalized_e)

        # 2. 计算KL散度
        kls = []
        for i in range(self.num_modalities):
            kl_sum = 0
            for j in range(self.num_modalities):
                if i != j:
                    kl = F.kl_div(
                        torch.log(probs_list[j] + self.epsilon),
                        probs_list[i] + self.epsilon,
                        reduction='none'
                    ).sum(dim=1)
                    kl_sum += kl
            kls.append(kl_sum / (self.num_modalities - 1))

        # 归一化KL散度
        for kl in kls:
            mean_kl = torch.mean(kl)
            std_kl = torch.std(kl)
            normalized_kl = (kl - mean_kl) / (std_kl + self.epsilon)
            normalized_kls.append(normalized_kl)

        # 3. 计算混合不确定性
        for i in range(self.num_modalities):
            hybrid_uncertainty = (self.beta * normalized_entropies[i] +
                                  (1 - self.beta) * normalized_kls[i])
            uncertainties.append(hybrid_uncertainty)

        # 4. 计算融合权重 (不确定性越低，权重越高)
        uncertainties_stack = torch.stack(uncertainties, dim=1)  # [B, M]
        weights = F.softmax(-uncertainties_stack, dim=1)  # [B, M]

        return uncertainties, weights

    def forward(self, mri_features, pet_features, mri_logits, pet_logits):
        """
        Args:
            mri_features: MRI特征 [B, D]
            pet_features: PET特征 [B, D]
            mri_logits: MRI分类logits [B, C]
            pet_logits: PET分类logits [B, C]
        Returns:
            fused_features: 融合后的特征 [B, D*M]
            weighted_logits: 加权后的logits [B, C]
            weights: 各模态的权重 [B, M]
        """
        logits_list = [mri_logits, pet_logits]

        # 计算不确定性和权重
        uncertainties, weights = self.compute_hybrid_uncertainty(logits_list)

        # 特征拼接（保持原有结构）
        fused_features = torch.cat([mri_features, pet_features], dim=1)

        # 加权融合预测
        weighted_logits = sum(weights[:, i].unsqueeze(1) * logits_list[i]
                              for i in range(self.num_modalities))

        return fused_features, weighted_logits, weights


