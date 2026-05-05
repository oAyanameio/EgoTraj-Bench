"""上下文评分器模块

该模块实现了用于生成智能体锚点和场景锚点的 AnchorHead 类，
用于轨迹预测中的上下文特征提取。
"""

import torch.nn as nn


class AnchorHead(nn.Module):
    """锚点生成头
    
    生成智能体锚点和场景锚点，用于轨迹预测中的上下文特征提取。
    """
    def __init__(self, dim):
        """初始化 AnchorHead
        
        Args:
            dim: 特征维度
        """
        super().__init__()
        # 智能体锚点网络
        self.agent_anchor = nn.Sequential(
            nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        # 场景池化网络
        self.scene_pool = nn.Sequential(
            nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, encoder_feat, agent_mask, agent_score):
        """前向传播
        
        Args:
            encoder_feat: 编码器特征，形状为 [B, A, D]
            agent_mask: 智能体掩码，形状为 [B, A]
            agent_score: 智能体分数，形状为 [B, A]
        
        Returns:
            agent_anchor: 智能体锚点，形状为 [B, A, D]
            scene_anchor: 场景锚点，形状为 [B, D]
        """
        # 应用智能体分数作为门控
        if agent_score is not None:
            encoder_feat = encoder_feat * agent_score.unsqueeze(-1)
        # 生成智能体锚点
        agent_anchor = self.agent_anchor(encoder_feat)

        # 计算场景锚点
        mask = agent_mask.float()
        denom = mask.sum(dim=1, keepdim=True)
        assert denom.min() > 0, "No valid agents"
        # 加权平均池化
        pooled = (encoder_feat * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
        # 生成场景锚点
        scene_anchor = self.scene_pool(pooled)  # [B, D]

        return agent_anchor, scene_anchor
