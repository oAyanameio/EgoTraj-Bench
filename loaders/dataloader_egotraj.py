"""EgoTraj 数据集加载器模块

本模块负责加载和处理 EgoTraj 数据集，包括数据整理、轨迹旋转和数据归一化等功能。
"""

import os
import numpy as np
import math
import torch
from utils.normalization import normalize_min_max
from torch.nn import functional as F
from einops import rearrange


def seq_collate_egotraj(batch):
    """
    整理批量数据，处理不同数量的智能体
    
    1. 添加过去的真实轨迹：past_traj_gt, past_traj_gt_orig
    2. 添加过去的有效标记：past_traj_valid
    
    Args:
        batch: 数据批次，包含多个数据项
        
    Returns:
        dict: 整理后的数据字典
    """
    # 使用 zip(*batch) 将批次数据解包为各个字段
    # batch 是由多个样本组成的列表，每个样本是一个元组
    # zip(*batch) 相当于对 batch 中的每个元组的相同位置元素进行分组
    # 结果是将混合的数据按字段重新组织成独立的元组
    (
        index,              # 样本索引/标识符
        num_peds,           # 每个样本中的智能体（行人）数量
        past_traj,          # 过去轨迹（相对坐标，经过旋转处理）
        fut_traj,           # 未来轨迹（相对坐标）
        past_traj_valid,    # 过去轨迹的有效性标记（用于处理遮挡）
        past_traj_orig,     # 过去轨迹的原始绝对坐标
        fut_traj_orig,      # 未来轨迹的原始绝对坐标
        traj_vel,          # 轨迹速度信息
        past_traj_gt,       # 过去真实轨迹（用于双流模型的另一个流）
        past_traj_gt_orig,  # 过去真实轨迹的原始绝对坐标
        past_theta,         # 过去的航向角（用于轨迹旋转）
    ) = zip(*batch)

    # 计算批次中最大智能体数量
    batch_max_agents = max(num_peds)
    # 初始化各种数据列表
    mask_list = []
    past_traj_list = []
    fut_traj_list = []
    past_traj_orig_list = []
    fut_traj_orig_list = []
    traj_vel_list = []
    past_traj_gt_list = []
    past_traj_gt_orig_list = []
    past_traj_valid_list = []
    past_theta_list = []

    # 处理每个数据项
    for i in range(len(batch)):
        # 计算需要填充的智能体数量
        pad_len = batch_max_agents - num_peds[i]
        if pad_len > 0:
            # 填充数据
            past_traj_list.append(
                F.pad(past_traj[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            fut_traj_list.append(
                F.pad(fut_traj[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            past_traj_orig_list.append(
                F.pad(past_traj_orig[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            fut_traj_orig_list.append(
                F.pad(fut_traj_orig[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            traj_vel_list.append(
                F.pad(traj_vel[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            # 创建掩码
            mask_list.append(torch.cat([torch.ones(num_peds[i]), torch.zeros(pad_len)]))

            # 填充过去的真实轨迹
            past_traj_gt_list.append(
                F.pad(past_traj_gt[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            past_traj_gt_orig_list.append(
                F.pad(
                    past_traj_gt_orig[i], (0, 0, 0, 0, 0, 0, 0, pad_len), "constant", 0
                )
            )
            past_traj_valid_list.append(
                F.pad(past_traj_valid[i], (0, 0, 0, 0, 0, pad_len), "constant", 0)
            )
            past_theta_list.append(F.pad(past_theta[i], (0, pad_len), "constant", 0))
        else:
            # 不需要填充，直接添加
            past_traj_list.append(past_traj[i])
            fut_traj_list.append(fut_traj[i])
            past_traj_orig_list.append(past_traj_orig[i])
            fut_traj_orig_list.append(fut_traj_orig[i])
            traj_vel_list.append(traj_vel[i])
            mask_list.append(torch.ones(num_peds[i]))
            past_traj_gt_list.append(past_traj_gt[i])
            past_traj_gt_orig_list.append(past_traj_gt_orig[i])
            past_traj_valid_list.append(past_traj_valid[i])
            past_theta_list.append(past_theta[i])

    # 堆叠数据
    indexes = torch.stack(index, dim=0)
    pre_motion_3D = torch.stack(past_traj_list, dim=0).squeeze(
        dim=2
    )  # [B, A_max, P, 6]
    fut_motion_3D = torch.stack(fut_traj_list, dim=0).squeeze(dim=2)
    pre_motion_3D_orig = torch.stack(past_traj_orig_list, dim=0).squeeze(dim=2)
    fut_motion_3D_orig = torch.stack(fut_traj_orig_list, dim=0).squeeze(dim=2)
    fut_traj_vel = torch.stack(traj_vel_list, dim=0).squeeze(dim=2)
    mask = torch.stack(mask_list, dim=0)
    batch_size = torch.tensor(pre_motion_3D.shape[0])  ### bt
    pre_motion_3D_gt = torch.stack(past_traj_gt_list, dim=0).squeeze(dim=2)
    pre_motion_3D_gt_orig = torch.stack(past_traj_gt_orig_list, dim=0).squeeze(dim=2)
    past_traj_valid = torch.stack(past_traj_valid_list, dim=0).squeeze(
        dim=2
    )  # [B, A_max, P]
    past_theta = torch.stack(past_theta_list, dim=0)  # [B, A_max]

    # 构建数据字典
    data = {
        "indexes": indexes,
        "batch_size": batch_size,
        "past_traj": pre_motion_3D,
        "fut_traj": fut_motion_3D,
        "past_traj_original_scale": pre_motion_3D_orig,
        "fut_traj_original_scale": fut_motion_3D_orig,
        "fut_traj_vel": fut_traj_vel,             # 未来轨迹的速度信息
        "agent_mask": mask,                       # 智能体掩码（标记真实智能体vs填充位置）
        "past_traj_gt": pre_motion_3D_gt,         # 过去真实轨迹（归一化后，用于双流模型的另一流）
        "past_traj_gt_original_scale": pre_motion_3D_gt_orig,  # 过去真实轨迹（原始尺度）
        "past_traj_valid": past_traj_valid,       # 过去轨迹的有效性标记（处理遮挡）
        "past_theta": past_theta,                 # 过去的航向角（用于轨迹旋转）
    }

    return data


def rotate_traj(
    past_rel,
    future_rel,
    past_abs,
    past_rel_gt,
    past_abs_gt,
    agents=2,
    rotate_time_frame=0,
    subset="eth",
):
    """
    对轨迹进行旋转处理
    
    Args:
        past_rel: 过去的相对轨迹
        future_rel: 未来的相对轨迹
        past_abs: 过去的绝对轨迹
        past_rel_gt: 过去的真实相对轨迹
        past_abs_gt: 过去的真实绝对轨迹
        agents: 智能体数量
        rotate_time_frame: 旋转参考时间帧
        subset: 数据集子集
        
    Returns:
        tuple: 旋转后的轨迹数据
    """
    # 重排张量维度
    past_rel = rearrange(past_rel, "b a p d -> (b a) p d")  # [A, P, 2]
    past_abs = rearrange(past_abs, "b a p d -> (b a) p d")  # [A, P, 2]
    future_rel = rearrange(future_rel, "b a f d -> (b a) f d")  # [A, F, 2]
    past_rel_gt = rearrange(past_rel_gt, "b a p d -> (b a) p d")  # [A, P, 2]
    past_abs_gt = rearrange(past_abs_gt, "b a p d -> (b a) p d")  # [A, P, 2]

    def calculate_rotate_matrix(past_rel_reference, rotate_time_frame):
        """计算旋转矩阵
        
        Args:
            past_rel_reference: 参考轨迹
            rotate_time_frame: 旋转参考时间帧
            
        Returns:
            tuple: (旋转矩阵, 旋转角度)
        """
        # 获取参考时间帧的运动
        past_diff = past_rel_reference[:, rotate_time_frame]  # [A, 2]
        
        # 计算轨迹方向角度
        past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0] + 1e-5))
        # 修正象限歧义
        past_theta = torch.where(
            (past_diff[:, 0] < 0), past_theta + math.pi, past_theta
        )
        # 构建旋转矩阵
        rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
        rotate_matrix[:, 0, 0] = torch.cos(past_theta)
        rotate_matrix[:, 0, 1] = torch.sin(past_theta)
        rotate_matrix[:, 1, 0] = -torch.sin(past_theta)
        rotate_matrix[:, 1, 1] = torch.cos(past_theta)
        
        return rotate_matrix, past_theta

    # 计算旋转矩阵和角度
    rotate_matrix, past_theta = calculate_rotate_matrix(
        past_rel, rotate_time_frame
    )

    # 应用旋转到所有轨迹
    past_after = torch.matmul(rotate_matrix, past_rel.transpose(1, 2)).transpose(1, 2)
    future_after = torch.matmul(rotate_matrix, future_rel.transpose(1, 2)).transpose(
        1, 2
    )
    past_abs_after = torch.matmul(rotate_matrix, past_abs.transpose(1, 2)).transpose(
        1, 2
    )
    past_abs_gt_after = torch.matmul(
        rotate_matrix, past_abs_gt.transpose(1, 2)
    ).transpose(1, 2)
    past_rel_gt_after = torch.matmul(
        rotate_matrix, past_rel_gt.transpose(1, 2)
    ).transpose(1, 2)

    # 恢复原始维度
    past_after = rearrange(past_after, "(b a) p d -> b a p d", a=agents)
    future_after = rearrange(future_after, "(b a) f d -> b a f d", a=agents)
    past_abs_after = rearrange(past_abs_after, "(b a) p d -> b a p d", a=agents)
    past_abs_gt_after = rearrange(past_abs_gt_after, "(b a) p d -> b a p d", a=agents)
    past_rel_gt_after = rearrange(past_rel_gt_after, "(b a) p d -> b a p d", a=agents)

    return (
        past_after,
        future_after,
        past_abs_after,
        past_abs_gt_after,
        past_rel_gt_after,
        past_theta,  # [A]
    )


class EgoTrajDataset(object):
    """EgoTraj 数据集类
    
    负责加载和处理 EgoTraj 数据集，包括数据加载、轨迹计算、旋转和归一化等功能。
    """
    
    def __init__(
        self,
        cfg=None,
        split="train",
        data_dir=None,
        rotate_time_frame=0,
        type="original",
        source="tbd",
    ):
        """
        初始化 EgoTrajDataset
        
        Args:
            cfg: 配置对象
            split: 数据集分割（train/test/val）
            data_dir: 数据目录
            rotate_time_frame: 旋转参考时间帧
            type: 数据类型
            source: 数据来源
        """
        # 根据类型和来源解析数据文件路径
        if type == "original" and source == "tbd":
            data_file_path = os.path.join(data_dir, f"egotraj_tbd_{split}.npz")
        elif type == "original_bal" and source != "tbd":
            data_file_path = os.path.join(data_dir, f"t2fpv_{source}_{split}.npz")
        else:
            raise ValueError(f"Invalid type/source combination: type={type}, source={source}")

        # 加载数据
        dset = np.load(data_file_path)
        self.all_obs = torch.from_numpy(dset["all_obs"])
        self.all_obs = self.all_obs[:, None, :, :]  # [N, 1, 8, 7]
        self.all_pred = torch.from_numpy(dset["all_pred"])
        self.all_pred = self.all_pred[:, None, :, :]  # [N, 1, 20, 7]
        self.num_peds_in_seq = torch.from_numpy(dset["num_peds"])  # [n_seq]
        self.seq_start_end = torch.from_numpy(dset["seq_start_end"])  # [n_seq, 2]

        self.cfg = cfg
        self.rotate_time_frame = rotate_time_frame
        self.split = split

        # 设置配置中的智能体数量
        max_agents = max(self.num_peds_in_seq)
        assert (
            cfg.MODEL.CONTEXT_ENCODER.AGENTS >= max_agents
        ), f"cfg.MODEL.CONTEXT_ENCODER.AGENTS: {cfg.MODEL.CONTEXT_ENCODER.AGENTS} < max_agents in {max_agents}"

        # 为轨迹旋转设置智能体数量
        cfg.agents = self.all_obs.shape[1]

        # 验证过去帧数量是否匹配
        assert self.all_obs.shape[2] == cfg.past_frames

        # 计算过去和未来轨迹
        past_traj_abs = self.all_obs[:, :, :, :2]  # [A, 1, P, 2]
        initial_pos = past_traj_abs[:, :, -1:]  # [A, 1, 1, 2]
        past_traj_rel = (past_traj_abs - initial_pos).contiguous()
        fut_traj = (
            self.all_pred[:, :, cfg.past_frames :, :2] - initial_pos
        ).contiguous()  # 相对未来轨迹

        # 添加过去的真实轨迹
        past_traj_abs_gt = self.all_pred[:, :, : cfg.past_frames, :2]  # [A, 1, P, 2]
        initial_pos_gt = past_traj_abs_gt[:, :, -1:]  # [A, 1, 1, 2]
        past_traj_rel_gt = (past_traj_abs_gt - initial_pos_gt).contiguous()
        # 列名: [x, y, orient/yaw, img_x, img_y, valid, agent_id]
        self.past_traj_valid = self.all_obs[:, :, :, 5]

        # 旋转轨迹（如果配置了旋转）
        if cfg.rotate:  # the normalization
            (
                past_traj_rel,
                fut_traj,
                past_traj_abs,
                past_traj_abs_gt,
                past_traj_rel_gt,
                past_theta,
            ) = rotate_traj(
                past_rel=past_traj_rel,
                future_rel=fut_traj,
                past_abs=past_traj_abs,
                past_rel_gt=past_traj_rel_gt,
                past_abs_gt=past_traj_abs_gt,
                agents=cfg.agents,
                rotate_time_frame=rotate_time_frame,
                subset=None,
            )

        # 保存旋转角度用于可视化
        if split == "test":
            self.past_theta = past_theta

        # 计算轨迹速度
        past_traj_vel = torch.cat(
            (
                past_traj_rel[:, :, 1:] - past_traj_rel[:, :, :-1],
                torch.zeros_like(past_traj_rel[:, :, -1:]),
            ),
            dim=2,
        )
        past_traj_vel_gt = torch.cat(
            (
                past_traj_rel_gt[:, :, 1:] - past_traj_rel_gt[:, :, :-1],
                torch.zeros_like(past_traj_rel_gt[:, :, -1:]),
            ),
            dim=2,
        )
        # 合并轨迹信息
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)
        past_traj_gt = torch.cat(
            (past_traj_abs_gt, past_traj_rel_gt, past_traj_vel_gt), dim=-1
        )
        # 计算未来轨迹速度
        self.fut_traj_vel = torch.cat(
            (
                fut_traj[:, :, 1:] - fut_traj[:, :, :-1],
                torch.zeros_like(fut_traj[:, :, -1:]),
            ),
            dim=2,
        )

        # 计算并保存归一化统计信息
        if split == "train":
            cfg.yml_dict["fut_traj_max"] = fut_traj.max().item()
            cfg.yml_dict["fut_traj_min"] = fut_traj.min().item()
            cfg.yml_dict["past_traj_max"] = past_traj.max().item()
            cfg.yml_dict["past_traj_min"] = past_traj.min().item()
            cfg.yml_dict["past_traj_gt_max"] = past_traj_gt.max().item()
            cfg.yml_dict["past_traj_gt_min"] = past_traj_gt.min().item()
        elif cfg.get("past_traj_min", None) is None:
            # 当训练统计信息不可用时，从当前数据计算
            cfg.yml_dict["fut_traj_max"] = fut_traj.max().item()
            cfg.yml_dict["fut_traj_min"] = fut_traj.min().item()
            cfg.yml_dict["past_traj_max"] = past_traj.max().item()
            cfg.yml_dict["past_traj_min"] = past_traj.min().item()
            cfg.yml_dict["past_traj_gt_max"] = past_traj_gt.max().item()
            cfg.yml_dict["past_traj_gt_min"] = past_traj_gt.min().item()

        # 记录原始数据以避免数值误差
        self.past_traj_original_scale = past_traj
        self.fut_traj_original_scale = fut_traj
        self.past_traj_gt_original_scale = past_traj_gt

        # 执行归一化
        if cfg.data_norm == "min_max":
            self.past_traj = normalize_min_max(
                past_traj, cfg.past_traj_min, cfg.past_traj_max, -1, 1
            ).contiguous()  # [A, 1, P, 6]
            self.fut_traj = normalize_min_max(
                fut_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
            ).contiguous()  # [A, 1, F, 2]
            self.past_traj_gt = normalize_min_max(
                past_traj_gt, cfg.past_traj_gt_min, cfg.past_traj_gt_max, -1, 1
            ).contiguous()  # [A, 1, P, 6]
        elif cfg.data_norm == "original":
            self.past_traj = past_traj
            self.fut_traj = fut_traj
            self.past_traj_gt = past_traj_gt

    def __len__(self):
        """返回数据集长度
        
        Returns:
            int: 数据集长度
        """
        return len(self.num_peds_in_seq)

    def __getitem__(self, item):
        """获取数据项
        
        Args:
            item: 数据索引
            
        Returns:
            list: 数据项列表
        """
        seq_start, seq_end = self.seq_start_end[item]
        num_peds = self.num_peds_in_seq[item]

        # 获取归一化和原始尺度的轨迹数据
        past_traj_norm_scale = self.past_traj[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate; norm
        fut_traj_norm_scale = self.fut_traj[
            seq_start:seq_end
        ]  # [A, F, 2] just rel; rotate; norm
        past_traj_original_scale = self.past_traj_original_scale[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate
        fut_traj_original_scale = self.fut_traj_original_scale[
            seq_start:seq_end
        ]  # [A, F, 2] just rel; rotate
        fut_traj_vel = self.fut_traj_vel[
            seq_start:seq_end
        ]  # [A, F, 2] just vel; rotate

        # 获取真实轨迹数据
        past_traj_gt_norm_scale = self.past_traj_gt[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate; norm
        past_traj_gt_original_scale = self.past_traj_gt_original_scale[
            seq_start:seq_end
        ]  # [A, P, 6] abs + rel + vel; rotate
        past_traj_valid = self.past_traj_valid[seq_start:seq_end]

        # 获取旋转角度
        if self.split == "test":
            past_theta = self.past_theta[seq_start:seq_end]
        else:
            past_theta = torch.zeros(
                past_traj_norm_scale.size(0), device=past_traj_norm_scale.device
            )  # dummy variable

        # 构建输出列表
        out = [
            torch.Tensor([item]).to(torch.int32),
            torch.Tensor([num_peds]).to(torch.int32),
            past_traj_norm_scale,  # [A, P, 6] -> eth A = 1
            fut_traj_norm_scale,
            past_traj_valid,
            past_traj_original_scale,
            fut_traj_original_scale,
            fut_traj_vel,
            past_traj_gt_norm_scale,
            past_traj_gt_original_scale,
            past_theta,
        ]
        return out
