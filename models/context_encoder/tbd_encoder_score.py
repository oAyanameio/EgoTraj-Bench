"""TBD 上下文编码器模块（带分数）

该模块实现了用于轨迹预测的上下文编码器，包括社交上下文编码和位置编码。
主要包含两个类：
- SocialTransformerScore：使用 Transformer 编码社交上下文信息
- ContextEncoderScore：整合场景和分数信息的上下文编码器
"""

import numpy as np
import torch
import torch.nn as nn
import math
from einops import rearrange


from models.utils import polyline_encoder
from models.utils.common_layers import SinusoidalPosEmb


class SocialTransformerScore(nn.Module):
    """社交 Transformer 编码器（带分数）
    
    使用 Transformer 编码社交上下文信息，并结合智能体分数作为门控。
    """
    def __init__(self, in_dim=48, hidden_dim=256, out_dim=128):
        """初始化 SocialTransformerScore
        
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出特征维度
        """
        super(SocialTransformerScore, self).__init__()
        # 线性层，将输入特征映射到隐藏层维度
        self.encode_past = nn.Linear(in_dim, hidden_dim, bias=False)
        # Transformer 编码器层
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True
        )
        # Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        # 输出线性层
        self.mlp_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, past_traj, mask, agent_score):
        """前向传播
        
        Args:
            past_traj: 过去轨迹，形状为 [B, A, P, D]
            mask: 智能体掩码，形状为 [B, A]，用于 padding
            agent_score: 智能体分数，形状为 [B, A]
        
        Returns:
            编码后的特征，形状为 [B, A, D]
        """
        B, A, P, D = past_traj.shape
        
        # 重塑轨迹数据：[B, A, P, D] -> [B, A, P*D]
        past_traj = rearrange(past_traj, "b a p d -> b a (p d)")
        # 编码轨迹特征
        h_feat = self.encode_past(past_traj)  # [B, A, D]

        # 应用智能体分数作为门控
        if agent_score is not None:
            h_feat = h_feat * agent_score.unsqueeze(-1)

        # 通过 Transformer 编码器
        h_feat_ = self.transformer_encoder(h_feat, src_key_padding_mask=mask)

        # 残差连接
        h_feat = h_feat + h_feat_
        # 输出层
        h_feat = self.mlp_out(h_feat)  # [B, A, D]

        return h_feat


class ContextEncoderScore(nn.Module):
    """上下文编码器（同时处理场景和分数）
    
    整合社交上下文信息和位置编码，为轨迹预测提供丰富的上下文特征。
    """

    def __init__(self, config, use_pre_norm):
        """初始化 ContextEncoderScore
        
        Args:
            config: 模型配置对象
            use_pre_norm: 是否使用预归一化
        """
        super().__init__()
        self.model_cfg = config
        dim = self.model_cfg.D_MODEL

        # 构建社交编码器
        # 输入是展平的 [P, D]，来自 past_traj，P=8, D=8 => 64
        self.agent_social_encoder = SocialTransformerScore(
            in_dim=64, hidden_dim=256, out_dim=dim
        )

        # 位置编码
        self.pos_encoding = nn.Sequential(
            SinusoidalPosEmb(dim, theta=10000),  # 正弦位置编码
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        # 智能体查询嵌入
        self.agent_query_embedding = nn.Embedding(self.model_cfg.AGENTS, dim)
        # 位置编码 MLP
        self.mlp_pe = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        # 构建 Transformer 编码器层
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            dropout=self.model_cfg.get("DROPOUT_OF_ATTN", 0.1),
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dim_feedforward=dim * 4,
            norm_first=use_pre_norm,
            batch_first=True,
        )
        # Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer, num_layers=self.model_cfg.NUM_ATTN_LAYERS
        )
        # 输出通道数
        self.num_out_channels = dim

    def forward(self, past_traj, agent_mask, agent_score):
        """前向传播
        
        Args:
            past_traj: 过去轨迹，形状为 [B, A, P, 6]
            agent_mask: 智能体掩码，形状为 [B, A]，1 表示有效，0 表示无效
            agent_score: 智能体分数，形状为 [B, A]
        
        Returns:
            编码后的上下文特征，形状为 [B, A, D]
        """

        B, A, P, D = past_traj.shape
        # 使用社交编码器编码智能体特征
        agent_feature = self.agent_social_encoder(
            past_traj=past_traj, mask=(agent_mask == 0), agent_score=agent_score
        )  # [B, A, D]

        # 计算位置编码
        pos_encoding = self.pos_encoding(torch.arange(A).to(past_traj.device))  # [A, D]

        # 计算智能体查询嵌入
        agent_query = self.agent_query_embedding(
            torch.arange(A).to(past_traj.device)
        )  # [A, D]

        # 融合查询嵌入和位置编码
        pos_encoding = self.mlp_pe(
            torch.cat([agent_query, pos_encoding], dim=-1)
        )  # [A, D]

        # 将位置编码添加到智能体特征
        agent_feature += pos_encoding.unsqueeze(0)  # [B, A, D]
        # 通过 Transformer 编码器
        # src_key_padding_mask: 忽略 padding 位置的 token
        encoder_out = self.transformer_encoder(
            agent_feature, src_key_padding_mask=(agent_mask == 0)
        )  # [B, A, D]

        return encoder_out
