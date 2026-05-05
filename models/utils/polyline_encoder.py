"""折线编码器模块

该模块实现了基于 PointNet 的折线编码器，用于处理和编码折线特征。
引用自 Motion Transformer (MTR) 项目：https://arxiv.org/abs/2209.13508
"""

import torch
import torch.nn as nn
from ..utils import common_layers


class PointNetPolylineEncoder(nn.Module):
    """基于 PointNet 的折线编码器
    
    使用 PointNet 架构对折线进行编码，提取折线的全局特征。
    """
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        """初始化 PointNetPolylineEncoder
        
        Args:
            in_channels: 输入特征通道数
            hidden_dim: 隐藏层维度
            num_layers: 总层数
            num_pre_layers: 预处理 MLP 层数
            out_channels: 输出通道数，为 None 时不使用输出 MLP
        """
        super().__init__()
        # 预处理 MLP
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        # 主 MLP
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,  # 输入是局部特征和全局特征的拼接
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        # 输出 MLP
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """前向传播
        
        Args:
            polylines: 折线数据，形状为 (batch_size, num_polylines, num_points_each_polylines, C)
            polylines_mask: 折线掩码，形状为 (batch_size, num_polylines, num_points_each_polylines)
        
        Returns:
            编码后的折线特征，形状为 (batch_size, num_polylines, C)
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # 预处理 MLP
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # 获取全局特征（最大池化）
        pooled_feature = polylines_feature.max(dim=2)[0]
        # 将全局特征与局部特征拼接
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # 主 MLP
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # 最大池化获取折线级特征
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # 输出 MLP
        if self.out_mlps is not None:
            # 过滤无效的折线
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            # 重建输出特征
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers
