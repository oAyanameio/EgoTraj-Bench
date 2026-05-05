"""数据归一化和反归一化工具函数模块

本模块提供了多种数据归一化和反归一化的函数，用于处理不同类型的数据转换需求。
"""

import torch

### Helper functions for linear normalization and unnormalization
def normalize_to_neg_one_to_one(img):
    """将输入数据归一化到[-1, 1]范围
    
    Args:
        img: 输入数据张量
        
    Returns:
        归一化后的数据张量
    """
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """将[-1, 1]范围的数据反归一化到[0, 1]范围
    
    Args:
        t: 输入数据张量
        
    Returns:
        反归一化后的数据张量
    """
    return (t + 1) * 0.5


def normalize_min_max(t, min_val, max_val, a, b, identity=False):
    """将数据归一化到[a, b]范围
    
    Args:
        t: 输入张量
        min_val: 输入数据的最小值
        max_val: 输入数据的最大值
        a: 输出范围的最小值
        b: 输出范围的最大值
        identity: 是否保持原样，不进行归一化
        
    Returns:
        归一化后的数据张量
    """
    if identity:
        return t
    else:
        return (b - a) * (t - min_val)/(max_val - min_val) + a 

def unnormalize_min_max(t, min_val, max_val, a, b, identity=False):
    """将[a, b]范围的数据反归一化回[min_val, max_val]范围
    
    Args:
        t: 输入张量
        min_val: 原始数据的最小值
        max_val: 原始数据的最大值
        a: 输入数据的最小值（归一化后的范围）
        b: 输入数据的最大值（归一化后的范围）
        identity: 是否保持原样，不进行反归一化
        
    Returns:
        反归一化后的数据张量
    """
    if identity:
        return t
    else:
        return (t - a) * (max_val - min_val)/(b - a) + min_val


def normalize_sqrt(traj_data, a, b):
    """使用平方根对轨迹数据进行归一化到[-1, 1]范围
    
    Args:
        traj_data: 轨迹数据张量，形状为[*, 2]
        a: 归一化参数，形状为[2]
        b: 归一化参数，形状为[2]
        
    Returns:
        归一化后的轨迹数据张量
    """
    traj_data = torch.abs(traj_data).sqrt() * torch.sign(traj_data)
    traj_data = traj_data / a.reshape(*([1] * (traj_data.dim() - 1)), -1) + b.reshape(*([1] * (traj_data.dim() - 1)), -1)
    return traj_data

def unnormalize_sqrt(traj_data, a, b):
    """使用平方根对轨迹数据进行反归一化
    
    Args:
        traj_data: 归一化后的轨迹数据张量，形状为[*, 2]
        a: 归一化参数，形状为[2]
        b: 归一化参数，形状为[2]
        
    Returns:
        反归一化后的轨迹数据张量
    """
    traj_data = (traj_data - b.reshape(*([1] * (traj_data.dim() - 1)), -1)) * a.reshape(*([1] * (traj_data.dim() - 1)), -1)
    traj_data = torch.sign(traj_data) * traj_data ** 2
    return traj_data
