"""流匹配模型模块

本模块实现了流匹配相关的功能，包括 FlowMatcher 基类和 BiFlowMatcher 子类，用于轨迹预测任务。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

from collections import namedtuple

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm
from utils.common import default
from utils.normalization import unnormalize_min_max
from utils.utils import LossBuffer


def pad_t_like_x(t, x):
    """将时间步张量填充为与输入张量相同的维度
    
    Args:
        t: 时间步张量
        x: 输入张量
        
    Returns:
        填充后的时间步张量
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FlowMatcher(nn.Module):
    """流匹配基类
    
    实现了流匹配的基本功能，包括噪声采样、前向传播和损失计算等。
    """
    
    def __init__(self, cfg, model, logger):
        """
        初始化 FlowMatcher
        
        Args:
            cfg: 配置对象
            model: 模型对象
            logger: 日志记录器
        """
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self.out_dim = cfg.MODEL.MODEL_OUT_DIM
        self.objective = cfg.objective
        self.sampling_steps = cfg.sampling_steps
        self.solver = cfg.get("solver", "euler")

        assert cfg.objective in {"pred_vel", "pred_data"}

        self.loss_buffer = LossBuffer(t_min=0, t_max=1.0, num_time_steps=100)

    @property
    def device(self):
        """获取设备
        
        Returns:
            模型所在的设备
        """
        return self.cfg.device

    def get_precond_coef(self, t):
        """获取预条件系数
        
        Args:
            t: 时间步
            
        Returns:
            tuple: (alpha_t, beta_t) 预条件系数
        """
        coef_1 = t.pow(2) * self.cfg.sigma_data**2 + (1 - t).pow(2)
        alpha_t = t * self.cfg.sigma_data**2 / coef_1
        beta_t = (1 - t) * self.cfg.sigma_data / coef_1.sqrt()
        return alpha_t, beta_t

    def get_input_scaling(self, t):
        """获取输入缩放因子
        
        Args:
            t: 时间步
            
        Returns:
            输入缩放因子
        """
        var_x_t = self.cfg.sigma_data**2 * t.pow(2) + (1 - t).pow(2)
        return 1.0 / var_x_t.sqrt().clip(min=1e-4, max=1e4)

    def fm_wrapper_func(self, x_t, t, model_out):
        """流匹配包装函数
        
        Args:
            x_t: 时间步 t 的潜在变量
            t: 时间步
            model_out: 模型输出
            
        Returns:
            包装后的模型输出
        """
        if self.cfg.fm_wrapper == "direct":
            return model_out
        elif self.cfg.fm_wrapper == "velocity":
            t = pad_t_like_x(t, x_t)
            return x_t + (1 - t) * model_out
        elif self.cfg.fm_wrapper == "precond":
            t = pad_t_like_x(t, x_t)
            alpha_t, beta_t = self.get_precond_coef(t)
            return alpha_t * x_t + beta_t * model_out

    def predict_vel_from_data(self, x1, xt, t):
        """从数据预测速度
        
        Args:
            x1: 目标数据
            xt: 时间步 t 的数据
            t: 时间步
            
        Returns:
            预测的速度
        """
        t = pad_t_like_x(t, x1)
        v = (x1 - xt) / (1 - t)
        return v

    def predict_data_from_vel(self, v, xt, t):
        """从速度预测数据
        
        Args:
            v: 速度
            xt: 时间步 t 的数据
            t: 时间步
            
        Returns:
            预测的数据
        """
        t = pad_t_like_x(t, xt)
        x1 = xt + v * (1 - t)
        return x1

    def fwd_sample_t(self, x0, x1, t):
        """前向采样时间步 t 的数据
        
        Args:
            x0: 初始噪声
            x1: 目标数据
            t: 时间步
            
        Returns:
            tuple: (xt, ut) 时间步 t 的数据和速度
        """
        t = pad_t_like_x(t, x0)
        xt = t * x1 + (1 - t) * x0
        ut = x1 - x0
        return xt, ut

    def get_reweighting(self, t, wrapper=None):
        """获取重加权因子
        
        Args:
            t: 时间步
            wrapper: 包装函数类型
            
        Returns:
            重加权因子
        """
        wrapper = default(wrapper, self.cfg.fm_wrapper)
        if wrapper == "direct":
            l_weight = torch.ones_like(t)
        elif wrapper == "velocity":
            l_weight = 1.0 / (1 - t) ** 2
        elif wrapper == "precond":
            alpha_t, beta_t = self.get_precond_coef(t)
            l_weight = 1.0 / beta_t**2
        if self.cfg.fm_rew_sqrt:
            l_weight = l_weight.sqrt()
        l_weight = l_weight.clamp(min=1e-4, max=1e4)
        return l_weight


ModelPredBiFlow = namedtuple(
    "ModelPredBiFlow",
    [
        "pred_vel_y",  # 预测的未来轨迹速度
        "pred_data_y",  # 预测的未来轨迹数据
        "pred_vel_x",  # 预测的过去轨迹速度
        "pred_data_x",  # 预测的过去轨迹数据
        "pred_score_y",  # 未来轨迹的预测分数
        "pred_score_x",  # 过去轨迹的预测分数
    ],
)


class BiFlowMatcher(FlowMatcher):
    """
    双向流匹配器，同时处理过去和未来轨迹的流匹配
    """

    def __init__(self, cfg, model, logger):
        """
        初始化 BiFlowMatcher
        
        Args:
            cfg: 配置对象
            model: 模型对象
            logger: 日志记录器
        """
        super().__init__(cfg, model, logger)

    def model_predictions(self, y_t, x_t, x_data, t, flag_print):
        """获取模型预测
        
        Args:
            y_t: 未来轨迹的噪声样本
            x_t: 过去轨迹的噪声样本
            x_data: 数据字典
            t: 时间步
            flag_print: 是否打印信息
            
        Returns:
            ModelPredBiFlow: 模型预测结果
        """
        # 输入缩放
        if self.cfg.fm_in_scaling:  # whether to scale the input to the FlowMatcher
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t)
            x_t_in = x_t * pad_t_like_x(self.get_input_scaling(t), x_t)
        else:
            y_t_in = y_t
            x_t_in = x_t

        # 模型前向传播
        model_out_past, denoiser_cls_past, model_out_future, denoiser_cls_future = (
            self.model(y_t_in, t, x_t_in, t, x_data)
        )

        # 应用流匹配包装函数
        y_data_at_t = self.fm_wrapper_func(y_t, t, model_out_future)  # [B, K, A, F * D]
        x_data_at_t = self.fm_wrapper_func(x_t, t, model_out_past)  # [B, K, A, P * D]

        if self.objective == "pred_vel":
            raise NotImplementedError

        elif self.objective == "pred_data":
            this_t = round(t.unique().item(), 4)

            if flag_print:
                if this_t == 0.0:
                    self.logger.info("{}".format("-" * 50))
                self.logger.info("Sampling time step: {:.3f}".format(this_t))

            # 预测速度
            pred_vel_y = self.predict_vel_from_data(y_data_at_t, y_t, t)
            pred_vel_x = self.predict_vel_from_data(x_data_at_t, x_t, t)

        else:
            raise ValueError(f"unknown objective {self.objective}")

        return ModelPredBiFlow(
            pred_vel_y,
            y_data_at_t,
            pred_vel_x,
            x_data_at_t,
            denoiser_cls_future,
            denoiser_cls_past,
        )

    @torch.inference_mode()
    def bwd_sample_t(
        self,
        y_t: torch.tensor,
        x_t: torch.tensor,
        t: int,
        dt: float,
        x_data: dict,
        flag_print: bool = False,
    ):
        """反向采样时间步 t
        
        Args:
            y_t: 未来轨迹的当前状态
            x_t: 过去轨迹的当前状态
            t: 时间步
            dt: 时间步长
            x_data: 数据字典
            flag_print: 是否打印信息
            
        Returns:
            tuple: 采样结果
        """
        B, K, _, _ = y_t.shape

        # 创建批量时间步
        batched_t = torch.full((B,), t, device=self.device, dtype=torch.float)
        # 获取模型预测
        model_preds = self.model_predictions(y_t, x_t, x_data, batched_t, flag_print)

        # 计算下一步状态
        y_next = y_t + model_preds.pred_vel_y * dt
        x_next = x_t + model_preds.pred_vel_x * dt
        return (
            y_next,
            model_preds.pred_data_y,
            x_next,
            model_preds.pred_data_x,
            model_preds,
        )

    @torch.no_grad()
    def sample(self, x_data, num_trajs, return_all_states=False):
        """
        从去噪模型中采样
        
        Args:
            x_data: 数据字典
            num_trajs: 采样轨迹数量
            return_all_states: 是否返回所有状态
            
        Returns:
            tuple: 采样结果
        """
        # 开始于 y_T ~ N(0,I)，使用反向蒙特卡洛条件去噪轨迹
        assert (
            num_trajs == self.cfg.denoising_head_preds
        ), "num_trajs must be equal to denoising_head_preds = {}".format(
            self.cfg.denoising_head_preds
        )  # denoising_head_preds: 20
        y_pred_data, x_pred_data = None, None
        self.in_dim = self.cfg.MODEL.MODEL_IN_DIM

        # 获取批次大小和智能体数量
        batch_size = x_data["batch_size"]
        num_agents = x_data["past_traj"].shape[1]  # max_agent
        # 初始化噪声
        y_t = torch.randn(
            (batch_size, num_trajs, num_agents, self.out_dim), device=self.device
        )
        x_t = torch.randn(
            (batch_size, num_trajs, num_agents, self.in_dim), device=self.device
        )

        # 绑定噪声（如果配置）
        if self.cfg.tied_noise:
            y_t = y_t[:, :1].expand(-1, self.cfg.denoising_head_preds, -1, -1)
            x_t = x_t[:, :1].expand(-1, self.cfg.denoising_head_preds, -1, -1)

        # 采样循环
        y_data_at_t_ls = []
        t_ls = []
        y_t_ls = []

        x_data_at_t_ls = []
        x_t_ls = []

        # 确定采样时间步
        if self.solver == "euler":
            dt = 1.0 / self.sampling_steps
            t_ls = dt * np.arange(self.sampling_steps)  # like [0., 0.1, 0.2, ..., 0.9]
            dt_ls = dt * np.ones(self.sampling_steps)
        elif self.solver == "lin_poly":
            # 前半部分线性时间增长，步长较小
            # 后半部分多项式增长的步长
            # 有助于在信号较强的地方集中模型能力
            lin_poly_long_step = self.cfg.lin_poly_long_step
            lin_poly_p = self.cfg.lin_poly_p

            n_steps_lin = self.sampling_steps // 2
            n_steps_poly = self.sampling_steps - n_steps_lin

            dt_lin = 1.0 / lin_poly_long_step
            t_lin_ls = dt_lin * np.arange(n_steps_lin)

            def _polynomially_spaced_points(a, b, N, p=2):
                # 在区间 [a, b] 中生成 N 个点，间距由幂 p 决定
                points = [
                    a + (b - a) * ((i - 1) ** p) / ((N - 1) ** p)
                    for i in range(1, N + 1)
                ]
                return points

            t_poly_start = t_lin_ls[-1] + dt_lin
            t_poly_end = 1.0
            t_poly_ls_ = _polynomially_spaced_points(
                t_poly_start, t_poly_end, n_steps_poly + 1, p=lin_poly_p
            )
            dt_poly = np.diff(t_poly_ls_)

            dt_ls = np.concatenate([dt_lin * np.ones(n_steps_lin), dt_poly]).tolist()
            t_ls = np.concatenate([t_lin_ls, t_poly_ls_[:-1]]).tolist()

        else:
            raise NotImplementedError(f"Unknown solver: {self.solver}")

        # 定义要打印的时间步
        num_prints = 10
        if len(t_ls) > num_prints:
            print_times = t_ls[:: self.sampling_steps // num_prints]
            if t_ls[-1] not in print_times:
                print_times.append(t_ls[-1])
        else:
            print_times = t_ls

        # 执行采样
        for idx_step, (cur_t, cur_dt) in enumerate(zip(t_ls, dt_ls)):
            flag_print = cur_t in print_times
            y_t, y_pred_data, x_t, x_pred_data, model_preds = self.bwd_sample_t(
                y_t, x_t, cur_t, cur_dt, x_data, flag_print
            )
            y_data_at_t_ls.append(y_pred_data)  # 更新模型预测
            x_data_at_t_ls.append(x_pred_data)
            if return_all_states:
                y_t_ls.append(y_t)  # 更新 y 的状态
                x_t_ls.append(x_t)

        # 整理结果
        y_data_at_t_ls = torch.stack(y_data_at_t_ls, dim=1)  # [B, S, K, A, F * D]
        x_data_at_t_ls = None
        t_ls = torch.tensor(t_ls, device=self.device)  # [S]
        if return_all_states:
            y_t_ls = torch.stack(y_t_ls, dim=1)  # [B, S, K, A, F * D]
            x_t_ls = torch.stack(x_t_ls, dim=1)  # [B, S, K, A, P * D]

        return (
            y_t,
            y_data_at_t_ls,
            y_t_ls,
            x_t,
            x_data_at_t_ls,
            x_t_ls,
            t_ls,
            model_preds.pred_score_y,
            model_preds.pred_score_x,
        )

    def get_loss_input(self, y_start_k, approx_t=None):
        """
        为流匹配模型训练准备输入
        
        Args:
            y_start_k: 目标数据
            approx_t: 近似时间步
            
        Returns:
            tuple: (t, x_t, u_t, target, l_weight) 时间步、噪声样本、速度、目标和重加权因子
        """

        # 随机时间步注入噪声
        bs = y_start_k.shape[0]

        if approx_t is None:
            # 根据配置的时间步调度策略采样
            if self.cfg.t_schedule == "uniform":
                t = torch.rand((bs,), device=self.device)
            elif self.cfg.t_schedule == "logit_normal":
                # 注意：这是 logit-normal（不是 log-normal）
                mean_ = self.cfg.logit_norm_mean
                std_ = self.cfg.logit_norm_std
                t_normal_ = torch.randn((bs,), device=self.device) * std_ + mean_
                t = torch.sigmoid(t_normal_)
            else:
                if "==" in self.cfg.t_schedule:
                    # 常数时间步
                    t = float(self.cfg.t_schedule.split("==")[1]) * torch.ones(
                        (bs,), device=self.device
                    )
                else:
                    # 自定义两阶段均匀分布
                    # 例如，'t0.5_p0.3' 表示以 30% 的概率从 [0, 0.5] 均匀采样，以 70% 的概率从 [0.5, 1] 均匀采样
                    cutoff_t = float(self.cfg.t_schedule.split("_")[0][1:])
                    prob_1 = float(self.cfg.t_schedule.split("_")[1][1:])

                    t_1 = torch.rand((bs,), device=self.device) * cutoff_t
                    t_2 = cutoff_t + torch.rand((bs,), device=self.device) * (
                        1 - cutoff_t
                    )
                    rand_num = torch.rand((bs,), device=self.device)

                    t = t_1 * (rand_num < prob_1) + t_2 * (rand_num >= prob_1)
        else:
            # 围绕近似时间步采样
            sigma = self.cfg.approx_t_std
            t = (approx_t + sigma * torch.randn((bs,), device=self.device)).clamp(
                0.0, 1.0
            )

        assert t.min() >= 0 and t.max() <= 1

        # 噪声样本
        if self.cfg.tied_noise:
            noise = torch.randn_like(y_start_k[:, 0:1])  # [B, 1, T, D]
            noise = noise.expand(
                -1, self.cfg.denoising_head_preds, -1, -1
            )  # [B, K, T, D]
        else:
            noise = torch.randn_like(y_start_k)  # [B, K, T, D]

        # 在时间步 t 采样潜在空间
        x_t, u_t = self.fwd_sample_t(x0=noise, x1=y_start_k, t=t)  # [B, K, T, D] * 2

        # 确定目标
        if self.objective == "pred_data":
            target = y_start_k
        elif self.objective == "pred_vel":
            target = u_t
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # 获取重加权因子
        l_weight = self.get_reweighting(t)

        return t, x_t, u_t, target, l_weight

    def p_losses(self, x_data, log_dict=None):
        """
        去噪模型训练
        
        Args:
            x_data: 数据字典
            log_dict: 日志字典
            
        Returns:
            tuple: 损失值
        """

        # 初始化
        K = self.cfg.denoising_head_preds
        assert self.objective == "pred_data", "only pred_data is supported for now"

        # 前向过程创建噪声样本
        fut_traj_normalized = repeat(x_data["fut_traj"], "b a f d -> b k a (f d)", k=K)
        past_traj_normalized = repeat(
            x_data["past_traj"][:, :, :, :2], "b a p d -> b k a (p d)", k=K
        )

        # 获取损失输入
        t_fut, y_t, y_u_t, _, y_l_weight = self.get_loss_input(
            y_start_k=fut_traj_normalized, approx_t=None
        )
        # 添加过去采样器
        t_pst, x_t, x_u_t, _, x_l_weight = self.get_loss_input(
            y_start_k=past_traj_normalized, approx_t=t_fut
        )

        # 模型前向传播
        if self.cfg.fm_in_scaling:
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t_fut), y_t)
            x_t_in = x_t * pad_t_like_x(self.get_input_scaling(t_pst), x_t)
        else:
            y_t_in = y_t
            x_t_in = x_t

        # 输入丢弃（如果配置）
        if self.training and self.cfg.get("drop_method", None) == "input":
            assert (
                self.cfg.get("drop_logi_k", None) is not None
                and self.cfg.get("drop_logi_m", None) is not None
            )
            m, k = self.cfg.drop_logi_m, self.cfg.drop_logi_k
            p_m_fut = 1 / (1 + torch.exp(-k * (t_fut - m)))
            p_m_fut = p_m_fut[:, None, None, None]
            y_t_in = y_t_in.masked_fill(torch.rand_like(p_m_fut) < p_m_fut, 0.0)

            p_m_pst = 1 / (1 + torch.exp(-k * (t_pst - m)))
            p_m_pst = p_m_pst[:, None, None, None]
            x_t_in = x_t_in.masked_fill(torch.rand_like(p_m_pst) < p_m_pst, 0.0)

        # 模型预测
        model_out_pst, denoiser_cls_pst, model_out_fut, denoiser_cls_fut = self.model(
            y_t_in=y_t_in, t_fut=t_fut, x_t_in=x_t_in, t_pst=t_pst, x_data=x_data
        )  # [B, K, A, F * D] 预测速度向量
        # 估计真实值
        denoised_y = self.fm_wrapper_func(
            y_t, t_fut, model_out_fut
        )  # 估计未来轨迹
        denoised_x = self.fm_wrapper_func(
            x_t, t_pst, model_out_pst
        )  # 估计过去轨迹

        # 获取智能体掩码
        if self.cfg.get("use_ablation_dataset", False):
            B, A = x_data["past_traj"].shape[:2]
            agent_mask = torch.ones((B, A), device=past_traj_normalized.device)
        else:
            agent_mask = x_data["agent_mask"]

        # 获取过去轨迹的真实值
        if self.cfg.get("USE_CLEAN_HIST", False):
            past_traj_gt_normalized = repeat(
                x_data["past_traj_gt"][:, :, :, :2], "b a p d -> b k a (p d)", k=K
            )
        else:
            past_traj_gt_normalized = past_traj_normalized

        # 计算过去轨迹的损失
        loss_reg_p, loss_cls_p, loss_reg_vel_p, _ = self.compute_loss(
            denoised_data=denoised_x,
            denoiser_cls=denoiser_cls_pst,
            l_weight=x_l_weight,
            traj_normalized=past_traj_gt_normalized,
            agent_mask=agent_mask,
        )
        # 使用原始未来轨迹相对值
        loss_reg_f, loss_cls_f, loss_reg_vel_f, loss_reg_b_f = self.compute_loss(
            denoised_data=denoised_y,
            denoiser_cls=denoiser_cls_fut,
            l_weight=y_l_weight,
            traj_normalized=fut_traj_normalized,
            agent_mask=agent_mask,
        )

        # 获取损失权重
        weight_reg = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get("reg", 1.0)
        weight_cls = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get("cls", 1.0)
        weight_vel = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get("vel", 0.2)

        weight_branch_pst = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get("branch_past", 0.3)

        # 计算分支损失
        loss_branch_fut = (
            weight_reg * loss_reg_f.mean()
            + weight_cls * loss_cls_f.mean()
            + weight_vel * loss_reg_vel_f.mean()
        )

        loss_branch_pst = (
            weight_reg * loss_reg_p.mean()
            + weight_cls * loss_cls_p.mean()
            + weight_vel * loss_reg_vel_p.mean()
        )

        # 总损失
        loss = loss_branch_fut + weight_branch_pst * loss_branch_pst

        # 记录损失
        flag_reset = self.loss_buffer.record_loss(
            t_fut, loss_reg_b_f.detach(), epoch_id=log_dict["cur_epoch"]
        )
        if flag_reset:
            dict_loss_per_level = self.loss_buffer.get_average_loss()
            log_dict.update({"denoiser_loss_per_level": dict_loss_per_level})

        return (
            loss,
            loss_reg_f.mean(),
            loss_cls_f.mean(),
            loss_reg_vel_f.mean(),
            loss_reg_p.mean(),
            loss_cls_p.mean(),
            loss_reg_vel_p.mean(),
        )

    def forward(self, x, log_dict=None):
        """前向传播
        
        Args:
            x: 输入数据
            log_dict: 日志字典
            
        Returns:
            损失值
        """
        return self.p_losses(x, log_dict)

    def compute_loss(
        self,
        denoised_data,
        denoiser_cls,
        l_weight,
        traj_normalized,
        agent_mask,
    ):
        """计算损失
        
        Args:
            denoised_data: 去噪后的数据
            denoiser_cls: 去噪器分类输出
            l_weight: 重加权因子
            traj_normalized: 归一化的轨迹
            agent_mask: 智能体掩码
            
        Returns:
            tuple: 损失值
        """

        B, K, A, feat_dim = traj_normalized.shape
        T = feat_dim // 2
        assert feat_dim % 2 == 0
        assert (
            T == self.cfg.future_frames or T == self.cfg.past_frames
        ), "T must be equal to future_frames or past_frames"

        # 组件选择
        denoised_data = rearrange(denoised_data, "b k a (f d) -> b k a f d", f=T)
        traj_normalized = traj_normalized.view(B, K, A, T, 2)

        # 获取轨迹的最小值和最大值
        traj_min = (
            self.cfg.fut_traj_min
            if T == self.cfg.future_frames
            else self.cfg.past_traj_min
        )
        traj_max = (
            self.cfg.fut_traj_max
            if T == self.cfg.future_frames
            else self.cfg.past_traj_max
        )

        # 反归一化
        if self.cfg.get("data_norm", None) == "min_max":
            denoised_data_metric = unnormalize_min_max(
                denoised_data, traj_min, traj_max, -1, 1
            )  # [B, K, A, T, D]
            traj_normalized_metric = unnormalize_min_max(
                traj_normalized, traj_min, traj_max, -1, 1
            )  # [B, K, A, T, D]
        elif self.cfg.get("data_norm", None) == "original":
            denoised_data_metric = denoised_data
            traj_normalized_metric = traj_normalized
        else:
            raise ValueError(
                f"Unknown data normalization method: {self.cfg.get('data_norm', None)}"
            )

        denoised_data_metric_xy = denoised_data_metric
        loss_reg_vel = torch.zeros(1).to(self.device)

        # 计算去噪误差
        mask = repeat(agent_mask, "b a -> b k a t d", k=K, t=T, d=2)  # [B, K, A, T, D]
        denoising_error_per_agent = (
            denoised_data_metric_xy - traj_normalized_metric
        ) * mask
        denoising_error_per_agent = denoising_error_per_agent.view(B, K, A, T, 2).norm(
            dim=-1
        )  # [B, K, A, T]

        # 对智能体求平均
        mask = repeat(agent_mask, "b a -> b k a t", k=K, t=T)  # [B, K, A, T]
        denoising_error_per_scene = (denoising_error_per_agent * mask).sum(
            dim=-2
        ) / mask.sum(dim=-2)

        # 损失减少方法
        if self.cfg.get("LOSS_REG_REDUCTION", "mean") == "mean":
            denoising_error_per_scene = denoising_error_per_scene.mean(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.mean(dim=-1)
        elif self.cfg.get("LOSS_REG_REDUCTION", "mean") == "sum":
            denoising_error_per_scene = denoising_error_per_scene.sum(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.sum(dim=-1)
        else:
            raise ValueError(
                f"Unknown reduction method: {self.cfg.get('LOSS_REG_REDUCTION', 'mean')}"
            )

        # 场景级损失
        if self.cfg.LOSS_NN_MODE == "scene":
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B]
            loss_reg_b = denoising_error_per_scene.gather(
                1, selected_components[:, None]
            ).squeeze(1)

            mask = repeat(agent_mask, "b a -> b k a", k=K)  # [B, K, A]
            assert mask.sum(dim=-1).min() > 0, "mask must be non-zero"
            cls_logits = (denoiser_cls * mask).sum(dim=-1) / mask.sum(dim=-1)
            loss_cls_b = F.cross_entropy(
                input=cls_logits, target=selected_components, reduction="none"
            )  # [B]

        # 智能体级损失
        elif self.cfg.LOSS_NN_MODE == "agent":
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A]
            loss_reg_b = denoising_error_per_agent.gather(
                1, selected_components[:, None, :]
            ).squeeze(
                1
            )  # [B, A]

            cls_logits = rearrange(denoiser_cls, "b k a -> (b a) k")  # [B * A, K]
            cls_labels = selected_components.view(-1)  # [B * A]

            mask = agent_mask
            assert mask.sum(dim=-1).min() > 0, "mask must be non-zero"
            loss_reg_b = (loss_reg_b * mask).sum(dim=-1) / mask.sum(dim=-1)

            mask_flat = mask.view(-1).bool()
            loss_cls_valid = F.cross_entropy(
                input=cls_logits[mask_flat],
                target=cls_labels[mask_flat],
                reduction="none",
            )  # [B * A]
            batch_ids = (
                torch.arange(B)
                .unsqueeze(1)
                .expand(B, A)
                .reshape(-1)
                .to(mask_flat.device)
            )
            batch_ids = batch_ids[mask_flat]
            loss_cls_b = torch.zeros(B, device=mask.device)
            counts = torch.zeros(B, device=mask.device)
            loss_cls_b.index_add_(0, batch_ids, loss_cls_valid)
            counts = mask.sum(dim=-1)
            loss_cls_b = loss_cls_b / counts  # [B]

        elif self.cfg.LOSS_NN_MODE == "both":
            raise NotImplementedError(
                "Both of agent and scene mode is not supported for now"
            )

        # 旧的损失计算
        loss_reg = (loss_reg_b * l_weight).mean()  # scalar
        loss_cls = loss_cls_b.mean()

        return loss_reg, loss_cls, loss_reg_vel, loss_reg_b
