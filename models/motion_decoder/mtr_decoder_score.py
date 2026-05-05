"""运动解码器模块（带分数）

该模块实现了用于轨迹预测的运动解码器，包括多个 Transformer 编码器块，
支持时间调制、自注意力机制和锚点特征融合。
"""

import copy
import torch.nn as nn
import torch
from einops import rearrange, repeat

from models.utils.common_layers import modulate


class MotionDecoderScore(nn.Module):
    """运动解码器（同时处理场景和分数）
    
    使用 Transformer 编码器实现运动解码，支持时间调制和锚点特征融合。
    """

    def __init__(self, config, use_pre_norm, use_adaln=True, use_anchor=False):
        """初始化 MotionDecoderScore
        
        Args:
            config: 模型配置对象
            use_pre_norm: 是否使用预归一化
            use_adaln: 是否使用自适应层归一化
            use_anchor: 是否使用锚点特征
        """
        super().__init__()
        self.num_blocks = config.get("NUM_DECODER_BLOCKS", 2)  # 解码器块数量
        self.self_attn_K = nn.ModuleList([])  # K 维度自注意力层列表
        self.self_attn_A = nn.ModuleList([])  # A 维度自注意力层列表
        
        # 模板编码器层
        template_encoder = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            dropout=config.get("DROPOUT_OF_ATTN", 0.1),
            nhead=config.NUM_ATTN_HEAD,
            dim_feedforward=config.D_MODEL * 4,
            norm_first=use_pre_norm,
            batch_first=True,
        )
        
        self.use_adaln = use_adaln  # 是否使用自适应层归一化
        self.dim = config.D_MODEL  # 特征维度
        self.use_anchor = use_anchor  # 是否使用锚点特征

        # 自适应层归一化模块
        if use_adaln:
            template_adaln = nn.Sequential(
                nn.SiLU(), nn.Linear(config.D_MODEL, 2 * config.D_MODEL, bias=True)
            )
            self.t_adaLN = nn.ModuleList([])

        # 锚点特征自适应层归一化模块
        if use_anchor:
            template_ach_adaln = nn.Sequential(
                nn.LayerNorm(config.D_MODEL),
                nn.SiLU(),
                nn.Linear(config.D_MODEL, 2 * config.D_MODEL, bias=True),
            )
            self.ach_adaLN = nn.ModuleList([])

        # 构建解码器块
        for _ in range(self.num_blocks):
            self.self_attn_K.append(copy.deepcopy(template_encoder))
            self.self_attn_A.append(copy.deepcopy(template_encoder))

            if use_adaln:
                self.t_adaLN.append(copy.deepcopy(template_adaln))
                # 零初始化 adaln 参数
                nn.init.constant_(self.t_adaLN[-1][-1].weight, 0)
                nn.init.constant_(self.t_adaLN[-1][-1].bias, 0)

            if use_anchor:
                self.ach_adaLN.append(copy.deepcopy(template_ach_adaln))
                # 零初始化 adaln 参数
                nn.init.constant_(self.ach_adaLN[-1][-1].weight, 0)
                nn.init.constant_(self.ach_adaLN[-1][-1].bias, 0)

    def forward(
        self,
        query_token,
        time_emb=None,
        agent_mask=None,
        anchor_agent=None,
        anchor_scene=None,
    ):
        """前向传播
        
        Args:
            query_token: 查询令牌，形状为 [B, K, A, D]
            time_emb: 时间嵌入，形状为 [B, D]
            agent_mask: 智能体掩码，形状为 [B, A]
            anchor_agent: 智能体锚点特征，形状为 [B, A, D]
            anchor_scene: 场景锚点特征，形状为 [B, D]
        
        Returns:
            解码后的特征，形状为 [B, K, A, D]
        """
        B, K, A = query_token.shape[:3]
        cur_query = query_token

        # 处理锚点特征
        if self.use_anchor:
            if anchor_agent is None:
                anchor_agent = torch.zeros(B, A, self.dim, device=query_token.device)
            if anchor_scene is None:
                anchor_scene = torch.zeros(B, self.dim, device=query_token.device)

            # 应用智能体掩码
            anchor_agent = anchor_agent * agent_mask.unsqueeze(-1)
            # 扩展维度
            anchor_agent = repeat(anchor_agent, "b a d -> b k a d", k=K)
            anchor_scene = repeat(anchor_scene, "b d -> b k a d", k=K, a=A)
            # 融合锚点特征
            anchor_ = anchor_agent + anchor_scene

        # 扩展智能体掩码
        agent_mask = repeat(agent_mask, "b a -> (b k) a", k=K)

        # 遍历解码器块
        for i in range(self.num_blocks):
            # 时间调制
            if self.use_adaln:
                shift, scale = self.t_adaLN[i](time_emb).chunk(2, dim=-1)
                cur_query = modulate(cur_query, shift, scale)  # [B, K, A, D]

            # K-to-K 自注意力
            cur_query = rearrange(cur_query, "b k a d -> (b a) k d")
            cur_query = self.self_attn_K[i](cur_query)

            # A-to-A 自注意力，添加智能体掩码
            cur_query = rearrange(cur_query, "(b a) k d -> (b k) a d", b=B)
            
            # 锚点特征调制
            if self.use_anchor:
                cur_query = rearrange(cur_query, "(b k) a d -> b k a d", b=B)
                shift, scale = self.ach_adaLN[i](anchor_).chunk(2, dim=-1)
                cur_query = modulate(cur_query, shift, scale)  # [B, K, A, D]
                cur_query = rearrange(cur_query, "b k a d -> (b k) a d", b=B)

            # 应用自注意力
            cur_query = self.self_attn_A[i](
                cur_query, src_key_padding_mask=(agent_mask == 0)
            )

            # 重塑形状
            cur_query = rearrange(cur_query, "(b k) a d -> b k a d", b=B)

        return cur_query
