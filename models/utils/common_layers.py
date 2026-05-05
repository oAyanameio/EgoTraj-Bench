"""通用层模块

该模块实现了轨迹预测中常用的通用层，包括：
- 正弦位置编码
- 特征调制
- 多层感知机构建
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码
    
    实现了 Transformer 中使用的正弦位置编码，用于为序列添加位置信息。
    """
    def __init__(self, dim, theta=10000):
        """初始化 SinusoidalPosEmb
        
        Args:
            dim: 位置编码的维度
            theta: 位置编码的缩放因子
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 [B]
        
        Returns:
            位置编码，形状为 [B, dim]
        """
        device = x.device
        half_dim = self.dim // 2
        # 计算位置编码的频率
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 计算位置编码
        emb = x[:, None] * emb[None, :]
        # 拼接正弦和余弦部分
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    """特征调制
    
    对输入特征进行调制，应用位移和缩放。
    
    Args:
        x: 输入特征
        shift: 位移参数
        scale: 缩放参数
    
    Returns:
        调制后的特征
    """
    if len(x.shape) == 3 and len(shift.shape) == 2:
        # [B, K, D] + [B, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 3:
        # [B, K, D] + [B, K, D]
        return x * (1 + scale) + shift
    elif len(x.shape) == 4 and len(shift.shape) == 2:
        # [B, K, A, D] + [B, D]
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(
            1
        )
    elif len(x.shape) == len(shift.shape) == 4:
        # [B, K, A, D] + [B, K, A, D]
        return x * (1 + scale) + shift
    else:
        raise ValueError("Invalid shapes to modulate")


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    """构建多层感知机
    
    根据给定的参数构建多层感知机网络。
    
    Args:
        c_in: 输入通道数
        mlp_channels: MLP 各层的通道数列表
        ret_before_act: 是否在激活函数之前返回
        without_norm: 是否不使用 BatchNorm
    
    Returns:
        构建好的 MLP 网络
    """
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        # 最后一层且需要在激活前返回
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            # 构建层
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        nn.BatchNorm1d(mlp_channels[k]),
                        nn.ReLU(),
                    ]
                )
            # 更新输入通道数
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)
