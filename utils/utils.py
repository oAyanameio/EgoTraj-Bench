"""通用工具函数模块

本模块提供了多种实用工具函数，包括代码备份、配置日志记录、KDE 负对数似然计算、
掩码应用、随机种子设置、日志记录器创建和损失缓冲区类。
"""

import os
import git
import logging
import shutil
import torch
import numpy as np
import random
import glob
from torch import nn
from pathlib import Path
from easydict import EasyDict
from scipy.stats import gaussian_kde


def back_up_code_git(cfg, logger):
    """使用 git 备份代码
    
    保存版本控制信息并备份代码到配置目录下的 code_backup 文件夹。
    
    Args:
        cfg: 配置对象
        logger: 日志记录器
    """
    # 保存版本控制信息
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info("git hash: {}".format(sha))

    # 备份代码
    code_backup_dir = Path(cfg.cfg_dir) / "code_backup"
    shutil.rmtree(code_backup_dir, ignore_errors=True)  # 删除旧的代码备份
    code_backup_dir.mkdir(parents=True, exist_ok=True)
    # 要保存的目录
    dirs_to_save = ["cfg", "models", "trainer", "loaders", "utils", "scripts"]
    # 复制目录
    [
        shutil.copytree(
            os.path.join(cfg.ROOT, this_dir), os.path.join(code_backup_dir, this_dir)
        )
        for this_dir in dirs_to_save
    ]
    # 查找并复制所有 Python 文件
    all_py_files = glob.glob(os.path.join(cfg.ROOT, "*.py"), recursive=True)
    [
        shutil.copy2(
            py_file, os.path.join(code_backup_dir, os.path.relpath(py_file, cfg.ROOT))
        )
        for py_file in all_py_files
    ]
    logger.info("Code is backedup to {}".format(code_backup_dir))


def log_config_to_file(cfg, pre="cfg_yml", logger=None):
    """将配置信息记录到日志文件
    
    递归记录配置字典中的所有键值对，包括嵌套的 EasyDict 对象。
    
    Args:
        cfg: 配置字典
        pre: 配置前缀
        logger: 日志记录器
    """
    logger.info("{} Config {} details below {}".format("=" * 20, pre, "=" * 20))
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info("--- %s.%s = edict() ---" % (pre, key))
            log_config_to_file(cfg[key], pre=pre + "." + key, logger=logger)
            continue
        logger.info("%s.%s: %s" % (pre, key, val))
    logger.info("{} Config {} details above {}".format("-" * 20, pre, "-" * 20))


def compute_kde_nll(pred_trajs, gt_traj):
    """计算 KDE 负对数似然
    
    使用高斯 KDE 计算预测轨迹和真实轨迹之间的负对数似然。
    
    Args:
        pred_trajs: 预测轨迹，形状为 [B, K, T, 2]
        gt_traj: 真实轨迹，形状为 [B, T, 2]
        
    Returns:
        tuple: (总负对数似然, 每个时间步的负对数似然)
    """
    kde_ll = 0.0
    log_pdf_lower_bound = -20  # 对数概率密度的下界
    num_timesteps = gt_traj.shape[1]  # 时间步数
    num_batches = pred_trajs.shape[0]  # 批次大小
    kde_ll_time = np.zeros(num_timesteps)  # 每个时间步的对数似然

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                # 为当前时间步的预测轨迹创建 KDE
                kde = gaussian_kde(pred_trajs[batch_num, :, timestep].T)
                # 计算真实轨迹在 KDE 下的对数概率密度
                pdf = np.clip(
                    kde.logpdf(gt_traj[batch_num, timestep]),
                    a_min=log_pdf_lower_bound,
                    a_max=None,
                )[0]
                # 累加对数似然
                kde_ll += pdf / (num_timesteps)
                kde_ll_time[timestep] += pdf
            except np.linalg.LinAlgError:
                # 处理线性代数错误（如奇异矩阵）
                kde_ll = np.nan

    return -kde_ll, -kde_ll_time  # 返回负对数似然


def apply_mask(input_tensor, mask, sample_dim=False):
    """
    对输入张量应用掩码
    
    Args:
        input_tensor: 输入张量，形状为 [B, A, F, D], [B, A, D] 或 [B, K, A, F, D]
        mask: 掩码张量，形状为 [B, A]
        sample_dim: 维度 1 是否为样本数
        
    Returns:
        应用掩码后的张量
    """
    extend_dims = len(input_tensor.shape) - len(mask.shape)
    if sample_dim:
        # 如果有样本维度，在样本维度上扩展掩码
        mask = mask.unsqueeze(1)
        mask = mask[(...,) + (None,) * (extend_dims - 1)]
    else:
        # 否则直接扩展掩码到输入张量的维度
        mask = mask[(...,) + (None,) * extend_dims]
    # 使用掩码填充输入张量
    return input_tensor.masked_fill(mask, 0.0)


def set_random_seed(rand_seed):
    """设置随机种子
    
    为 NumPy、Python 随机模块、PyTorch 以及 CUDA 设置随机种子，
    确保实验的可重复性。
    
    Args:
        rand_seed: 随机种子值
    """
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    # 确保 CUDA 确定性
    torch.backends.cudnn.deterministic = True
    # 禁用 CUDA 基准测试以确保确定性
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    """创建日志记录器
    
    创建一个配置好的日志记录器，可同时输出到控制台和文件。
    
    Args:
        log_file: 日志文件路径
        rank: 进程排名，仅排名为 0 的进程输出日志
        log_level: 日志级别
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else "ERROR")
    # 日志格式
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    # 控制台处理器
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else "ERROR")
    console.setFormatter(formatter)
    logger.addHandler(console)
    # 文件处理器（如果提供了日志文件路径）
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else "ERROR")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 禁用日志传播
    logger.propagate = False
    return logger


class LossBuffer:
    """损失缓冲区类，用于记录和管理不同去噪级别的损失
    
    该类用于记录每个去噪级别的损失，并在每个 epoch 后重置。
    """
    
    def __init__(self, t_min, t_max, num_time_steps):
        """
        初始化 LossBuffer
        
        Args:
            t_min: 最小去噪级别
            t_max: 最大去噪级别
            num_time_steps: 去噪级别的数量
        """
        self.t_min = t_min
        self.t_max = t_max
        self.num_time_steps = num_time_steps
        # 生成均匀分布的去噪级别区间
        self.t_interval = np.linspace(t_min, t_max, num_time_steps)
        # 初始化每个去噪级别的损失数据列表
        self.loss_data = [[] for _ in range(self.num_time_steps)]
        self.last_epoch = -1

    def record_loss(self, t, loss, epoch_id):
        """
        记录特定去噪级别的损失
        
        Args:
            t: 去噪级别，形状为 [B]
            loss: 损失值，形状为 [B]
            epoch_id: 当前 epoch ID
            
        Returns:
            bool: 是否在新的 epoch 中重置了损失数据
        """

        flag_reset = False
        # 检查是否进入了新的 epoch
        if epoch_id != self.last_epoch:
            self.last_epoch = epoch_id
            self.reset()
            flag_reset = epoch_id > 0

        # 将张量转换为 NumPy 数组
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().numpy()

        # 确定每个损失值对应的去噪级别区间
        idx = np.digitize(t, self.t_interval) - 1
        # 记录损失值
        for i, l in zip(idx, loss):
            self.loss_data[i].append(l)

        return flag_reset

    def reset(self):
        """
        为新的 epoch 重置损失数据
        """
        self.loss_data = [[] for _ in range(self.num_time_steps)]

    def get_average_loss(self):
        """
        计算每个去噪级别的平均损失
        
        Returns:
            dict: 每个去噪级别对应的平均损失
        """
        # 计算每个去噪级别的平均损失
        avg_loss_per_level = [np.mean(l) if len(l) > 0 else 0.0 for l in self.loss_data]
        # 构建去噪级别到平均损失的映射
        dict_loss_per_level = {
            t: l for t, l in zip(self.t_interval, avg_loss_per_level)
        }
        return dict_loss_per_level
