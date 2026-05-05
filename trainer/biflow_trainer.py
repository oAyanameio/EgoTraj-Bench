"""BiFlow 模型训练器模块

该模块实现了 BiFlow 模型的训练、评估和测试功能，包括：
- 模型训练循环
- 学习率调度器构建
- 优化器构建
- 模型评估（ADE、FDE、JADE、JFDE 等指标计算）
- 模型测试和结果保存
- 模型 checkpoint 管理
- EMA（指数移动平均）模型维护

主要类和函数：
- BiFlowTrainer：BiFlow 模型的训练器类
- build_scheduler：构建学习率调度器
- build_optimizer：构建优化器
- cycle：创建数据加载器的循环迭代器
"""
import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from utils.common import exists, default
from utils.utils import set_random_seed
from utils.normalization import unnormalize_min_max


def cycle(dl):
    """创建数据加载器的循环迭代器
    
    用于在训练过程中无限循环遍历数据加载器，确保训练可以持续进行
    直到达到指定的训练步数。
    
    Args:
        dl (DataLoader): PyTorch 数据加载器
    
    Yields:
        batch: 数据加载器中的批次数据
    """
    while True:
        for data in dl:
            yield data


def build_scheduler(optimizer, opt_cfg, total_iters_each_epoch):
    """构建学习率调度器
    
    根据配置构建不同类型的学习率调度器，支持多种调度策略：
    - cosineAnnealingLRwithWarmup：带预热的余弦退火调度器
    - lambdaLR：自定义 lambda 函数调度器
    - linearLR：线性衰减调度器
    - stepLR：步进衰减调度器
    - cosineAnnealingLR：余弦退火调度器
    
    Args:
        optimizer (Optimizer): PyTorch 优化器
        opt_cfg (Config): 优化配置对象，包含学习率相关参数
        total_iters_each_epoch (int): 每个 epoch 的迭代次数
    
    Returns:
        scheduler (LR_scheduler): PyTorch 学习率调度器
    """
    total_epochs = opt_cfg.NUM_EPOCHS
    decay_steps = [
        x * total_iters_each_epoch
        for x in opt_cfg.get("DECAY_STEP_LIST", [5, 10, 15, 20])
    ]

    def lr_lbmd(cur_epoch):
        """Lambda 函数，用于计算学习率衰减因子"""
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * opt_cfg.LR_DECAY
        return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

    if opt_cfg.get("SCHEDULER", None) == "cosineAnnealingLRwithWarmup":
        # 带预热的余弦退火调度器
        total_iterations = total_epochs * total_iters_each_epoch
        warmup_iterations = max(1, int(total_iterations * 0.05))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: max(opt_cfg.LR_CLIP / opt_cfg.LR, step / warmup_iterations),
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iterations - warmup_iterations,
            eta_min=opt_cfg.LR_CLIP,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iterations],
        )
    elif opt_cfg.get("SCHEDULER", None) == "lambdaLR":
        # 自定义 lambda 函数调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
    elif opt_cfg.get("SCHEDULER", None) == "linearLR":
        # 线性衰减调度器
        total_iters = total_iters_each_epoch * total_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=opt_cfg.LR_CLIP / opt_cfg.LR,
            total_iters=total_iters,
        )
    elif opt_cfg.get("SCHEDULER", None) == "stepLR":
        # 步进衰减调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt_cfg.DECAY_STEP, gamma=opt_cfg.DECAY_GAMMA
        )
    elif opt_cfg.get("SCHEDULER", None) == "cosineAnnealingLR":
        # 余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=opt_cfg.LR_CLIP
        )
    else:
        # 无调度器
        scheduler = None
    return scheduler


def build_optimizer(model, opt_cfg):
    """构建优化器
    
    根据配置构建不同类型的优化器，支持 Adam 和 AdamW 优化器。
    
    Args:
        model (nn.Module): PyTorch 模型
        opt_cfg (Config): 优化配置对象，包含优化器相关参数
    
    Returns:
        optimizer (Optimizer): PyTorch 优化器
    """
    if opt_cfg.OPTIMIZER == "Adam":
        # Adam 优化器
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters()],
            lr=opt_cfg.LR,
            weight_decay=opt_cfg.get("WEIGHT_DECAY", 0),
        )
    elif opt_cfg.OPTIMIZER == "AdamW":
        # AdamW 优化器（带权重衰减）
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.LR,
            weight_decay=opt_cfg.get("WEIGHT_DECAY", 0),
        )
    else:
        # 不支持的优化器类型
        assert False
    return optimizer


class BiFlowTrainer(object):
    """BiFlow 模型训练器类
    
    负责 BiFlow 模型的训练、评估和测试，包括：
    - 模型训练循环
    - 模型评估（计算 ADE、FDE、JADE、JFDE 等指标）
    - 模型测试和结果保存
    - 模型 checkpoint 管理
    - EMA（指数移动平均）模型维护
    """
    def __init__(
        self,
        cfg,
        denoiser,
        train_loader,
        test_loader,
        val_loader=None,
        tb_log=None,
        logger=None,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
        save_samples=False,
        *awgs,
        **kwargs,
    ):
        """初始化 BiFlowTrainer
        
        Args:
            cfg (Config): 配置对象，包含模型和训练相关参数
            denoiser (nn.Module): BiFlow 模型的去噪器
            train_loader (DataLoader): 训练数据加载器
            test_loader (DataLoader): 测试数据加载器
            val_loader (DataLoader, optional): 验证数据加载器，默认为 None
            tb_log (SummaryWriter, optional): TensorBoard 日志记录器，默认为 None
            logger (Logger, optional): 日志记录器，默认为 None
            gradient_accumulate_every (int, optional): 梯度累积步数，默认为 1
            ema_decay (float, optional): EMA 衰减系数，默认为 0.995
            ema_update_every (int, optional): EMA 更新频率，默认为 1
            save_samples (bool, optional): 是否保存采样结果，默认为 False
        """
        super().__init__()

        # 初始化基本参数
        self.cfg = cfg
        self.denoiser = denoiser
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 如果没有提供验证数据加载器，则使用测试数据加载器
        self.val_loader = default(val_loader, test_loader)
        self.tb_log = tb_log
        self.logger = logger

        self.gradient_accumulate_every = gradient_accumulate_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every

        # 配置去噪相关参数
        if cfg.denoising_method == "fm":
            self.denoiser_steps = cfg.sampling_steps
            self.denoising_schedule = cfg.t_schedule
        else:
            raise NotImplementedError(
                f"Denoising method [{cfg.denoising_method}] is not implemented yet."
            )

        self.save_dir = Path(cfg.cfg_dir)

        # 采样和训练超参数
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_samples = save_samples

        # 初始化加速器
        self.accelerator = Accelerator(split_batches=True, mixed_precision="no")

        # 初始化 EMA 模型
        if self.accelerator.is_main_process:
            self.ema = EMA(denoiser, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        if train_loader is not None:
            # 计算保存和采样频率
            self.save_and_sample_every = cfg.checkpt_freq * len(train_loader)
            # 计算总训练步数
            self.train_num_steps = cfg.OPTIMIZATION.NUM_EPOCHS * len(train_loader)

            # 构建优化器和调度器
            self.opt = build_optimizer(self.denoiser, self.cfg.OPTIMIZATION)
            self.scheduler = build_scheduler(
                self.opt, self.cfg.OPTIMIZATION, len(self.train_loader)
            )

            # 使用 accelerator 准备模型、数据加载器和优化器
            self.denoiser, self.opt = self.accelerator.prepare(self.denoiser, self.opt)

            train_dl_ = self.accelerator.prepare(train_loader)
            self.train_loader = train_dl_
            # 创建循环数据迭代器
            self.dl = cycle(train_dl_)
        else:
            self.save_and_sample_every = 0
            self.train_num_steps = 1  # 避免在评估时除零错误
            self.denoiser = self.accelerator.prepare(self.denoiser)

        # 准备测试和验证数据加载器
        self.test_loader = self.accelerator.prepare(test_loader)
        val_loader = default(val_loader, test_loader)
        self.val_loader = self.accelerator.prepare(val_loader)

        # 设置计数器和训练状态
        self.step = 0
        self.best_ade_min = float("inf")

        # 打印模型参数数量
        self.print_model_params(self.denoiser, "Stage One Model")

    def print_model_params(self, model: nn.Module, name: str):
        """打印模型参数数量
        
        计算并打印模型的总参数数量和可训练参数数量。
        
        Args:
            model (nn.Module): PyTorch 模型
            name (str): 模型名称
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            f"[{name}] Trainable/Total Params: {trainable_num}/{total_num}"
        )

    @property
    def device(self):
        """获取设备信息
        
        返回模型运行的设备，从配置中获取。
        
        Returns:
            device (torch.device): 模型运行的设备
        """
        return self.cfg.device

    def save_ckpt(self, ckpt_name):
        """保存模型 checkpoint
        
        保存模型、优化器、EMA、调度器和缩放器的状态字典。
        
        Args:
            ckpt_name (str): checkpoint 名称
        """
        if not self.accelerator.is_local_main_process:
            return
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.denoiser),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
        }
        torch.save(data, os.path.join(self.cfg.model_dir, f"{ckpt_name}.pt"))

    def save_last_ckpt(self):
        """保存最后一个 checkpoint
        
        保存模型、优化器、EMA 和调度器的状态字典到 checkpoint_last.pt 文件。
        """
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.denoiser),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(data, os.path.join(self.cfg.model_dir, "checkpoint_last.pt"))

    def load(self, ckpt_name):
        """加载模型 checkpoint
        
        从指定的 checkpoint 文件加载模型、优化器、EMA、调度器和缩放器的状态。
        
        Args:
            ckpt_name (str): checkpoint 名称
        """
        accelerator = self.accelerator

        # 处理 epoch 数变化的情况
        new_epochs = self.cfg.OPTIMIZATION.NUM_EPOCHS
        old_epochs = self.cfg.RESUME.start_epoch
        model_dir = self.cfg.model_dir.replace(f"EP{new_epochs}", f"EP{old_epochs}")

        # 加载 checkpoint 数据
        data = torch.load(
            os.path.join(model_dir, f"{ckpt_name}.pt"),
            map_location=self.device,
            weights_only=True,
        )

        # 加载模型状态
        model = self.accelerator.unwrap_model(self.denoiser)
        model.load_state_dict(data["model"])

        # 加载训练状态
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        # 只有当 epoch 数不变时才加载调度器状态
        if new_epochs == old_epochs:
            self.scheduler.load_state_dict(data["scheduler"])

        # 加载 EMA 状态
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        # 打印版本信息
        if "version" in data:
            print(f"loading from version {data['version']}")

        # 加载缩放器状态
        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        """模型训练循环
        
        执行模型训练的主要逻辑，包括：
        - 加载 checkpoint（如果需要）
        - 训练循环
        - 梯度累积
        - 损失计算和反向传播
        - 模型评估
        - 模型保存
        """

        # 初始化
        accelerator = self.accelerator
        # 加载 checkpoint（如果需要）
        if self.cfg.RESUME.resume:
            ckpt_name = self.cfg.RESUME.get("ckpt_name", "checkpoint_last")
            self.load(ckpt_name)
            self.logger.info(f"Resuming training from {ckpt_name}.pt")

        self.logger.info("training start")
        # 计算每个 epoch 的迭代次数
        iter_per_epoch = self.train_num_steps // self.cfg.OPTIMIZATION.NUM_EPOCHS

        # 设置早停步数
        if self.cfg.RESUME.get("early_stop", -1) > 0:
            self.early_stop_num_steps = (
                self.cfg.RESUME.get("early_stop", -1) * iter_per_epoch
            )
        else:
            self.early_stop_num_steps = self.train_num_steps

        # 训练进度条
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while (
                self.step < self.train_num_steps
                and self.step < self.early_stop_num_steps
            ):
                # 初始化每迭代变量
                total_loss = 0.0
                self.denoiser.train()
                self.ema.ema_model.train()

                # 梯度累积
                for _ in range(self.gradient_accumulate_every):
                    # 获取批次数据
                    data = {k: v.to(self.device) for k, v in next(self.dl).items()}

                    log_dict = {"cur_epoch": self.step // iter_per_epoch}

                    # 计算损失
                    with self.accelerator.autocast():
                        (
                            loss,
                            loss_reg_fut,
                            loss_cls_fut,
                            _,
                            loss_reg_pst,
                            loss_cls_pst,
                            _,
                        ) = self.denoiser(data, log_dict)
                        # 梯度累积时，损失需要除以累积步数
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播
                    self.accelerator.backward(loss)

                    # 记录到 TensorBoard
                    if self.tb_log is not None:
                        self.tb_log.add_scalar(
                            "train/loss_total", loss.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_reg_fut", loss_reg_fut.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_cls_fut", loss_cls_fut.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_reg_pst", loss_reg_pst.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/loss_cls_pst", loss_cls_pst.item(), self.step
                        )
                        self.tb_log.add_scalar(
                            "train/learning_rate",
                            self.opt.param_groups[0]["lr"],
                            self.step,
                        )

                # 更新进度条描述
                pbar.set_description(
                    f'curr step: {self.step}/{self.train_num_steps}, total loss: {total_loss:.4f}, loss_reg_fut: {loss_reg_fut:.4f}, loss_cls_fut: {loss_cls_fut:.4f}, loss_reg_pst: {loss_reg_pst:.4f}, loss_cls_pst: {loss_cls_pst:.4f}, lr: {self.opt.param_groups[0]["lr"]:.6f}'
                )

                # 等待所有进程完成
                accelerator.wait_for_everyone()
                # 梯度裁剪
                accelerator.clip_grad_norm_(
                    self.denoiser.parameters(), self.cfg.OPTIMIZATION.GRAD_NORM_CLIP
                )

                # 更新参数
                self.opt.step()
                # 清零梯度
                self.opt.zero_grad()

                # 等待所有进程完成
                accelerator.wait_for_everyone()

                # 主进程执行的操作
                if accelerator.is_main_process:
                    # 更新 EMA 模型
                    self.ema.update()
                    # 检查点测试和保存最佳验证模型
                    if (self.step + 1) >= self.save_and_sample_every and (
                        self.step + 1
                    ) % self.save_and_sample_every == 0:

                        # 评估模型
                        fut_traj_gt, performance, n_samples = self.eval_dataloader(
                            testing_mode=False, training_err_check=False
                        )

                        # 计算当前 ADE_min
                        cur_epoch = self.step // iter_per_epoch
                        select_idx = len(self.cfg.K_LIST) - 1
                        cur_ade_min = performance["ADE_min"][select_idx] / n_samples

                        # 更新最佳模型
                        if cur_ade_min < self.best_ade_min:
                            self.best_ade_min = cur_ade_min
                            self.logger.info(
                                f"Current best ADE_MIN: {self.best_ade_min}"
                            )
                            self.save_ckpt(f"checkpoint_best")

                        # 保存模型并删除旧模型
                        ckpt_list = glob(
                            os.path.join(self.cfg.model_dir, "checkpoint_epoch_*.pt*")
                        )
                        ckpt_list.sort(key=os.path.getmtime)

                        if ckpt_list.__len__() >= self.cfg.max_num_ckpts:
                            for cur_file_idx in range(
                                0, len(ckpt_list) - self.cfg.max_num_ckpts + 1
                            ):
                                os.remove(ckpt_list[cur_file_idx])

                        # 保存当前 epoch 的模型
                        self.save_ckpt("checkpoint_epoch_%d" % cur_epoch)

                # 更新步数
                self.step += 1
                pbar.update(1)
                # 更新学习率
                self.scheduler.step()

                # 一次训练迭代结束
            # 训练循环结束

        # 保存最后一个 checkpoint
        self.save_last_ckpt()

        self.logger.info("training complete")

    def compute_ADE_FDE(self, distances, end_frame):
        """计算 ADE（平均位移误差）和 FDE（最终位移误差）
        
        Args:
            distances: 距离张量，形状为 [b*num_agents, k_preds, future_frames] 或 [b*num_agents, timestamps, k_preds, future_frames]
            end_frame: 结束帧索引
        
        Returns:
            ade_best: 最佳预测轨迹的 ADE
            fde_best: 最佳预测轨迹的 FDE
            ade_avg: 所有预测轨迹的平均 ADE
            fde_avg: 所有预测轨迹的平均 FDE
        """
        # 计算最佳预测轨迹的 ADE（平均位移误差）
        ade_best = (
            (distances[..., :end_frame]).mean(dim=-1).min(dim=-1).values.sum(dim=0)
        )
        # 计算最佳预测轨迹的 FDE（最终位移误差）
        fde_best = (distances[..., end_frame - 1]).min(dim=-1).values.sum(dim=0)
        # 计算所有预测轨迹的平均 ADE
        ade_avg = (distances[..., :end_frame]).mean(dim=-1).mean(dim=-1).sum(dim=0)
        # 计算所有预测轨迹的平均 FDE
        fde_avg = (distances[..., end_frame - 1]).mean(dim=-1).sum(dim=0)
        return ade_best, fde_best, ade_avg, fde_avg

    ### Based on https://arxiv.org/abs/2305.06292 Joint metric for ADE and FDE
    def compute_JADE_JFDE(self, distances, end_frame):
        """计算 JADE（联合平均位移误差）和 JFDE（联合最终位移误差）
        
        基于论文 "Joint metric for ADE and FDE" 实现的联合评估指标。
        
        Args:
            distances: 距离张量，形状为 [b*num_agents, k_preds, future_frames] 或 [b*num_agents, timestamps, k_preds, future_frames]
            end_frame: 结束帧索引
        
        Returns:
            jade_best: 最佳预测轨迹的 JADE
            jfde_best: 最佳预测轨迹的 JFDE
            jade_avg: 所有预测轨迹的平均 JADE
            jfde_avg: 所有预测轨迹的平均 JFDE
        """
        # 计算最佳预测轨迹的 JADE（联合平均位移误差）
        jade_best = (
            (distances[..., :end_frame]).mean(dim=-1).sum(dim=0).min(dim=-1).values
        )
        # 计算最佳预测轨迹的 JFDE（联合最终位移误差）
        jfde_best = (distances[..., end_frame - 1]).sum(dim=0).min(dim=-1).values
        # 计算所有预测轨迹的平均 JADE
        jade_avg = (distances[..., :end_frame]).mean(dim=-1).sum(dim=0).mean(dim=0)
        # 计算所有预测轨迹的平均 JFDE
        jfde_avg = (distances[..., end_frame - 1]).sum(dim=0).mean(dim=-1)
        return jade_best, jfde_best, jade_avg, jfde_avg

    def compute_avar_fvar(self, pred_trajs, end_frame):
        """计算 AVar（平均方差）和 FVar（最终方差）
        
        用于评估预测轨迹的多样性。
        
        Args:
            pred_trajs: 预测轨迹张量，形状为 [b*num_agents, k_preds, future_frames, dim]
            end_frame: 结束帧索引
        
        Returns:
            a_var: 平均方差
            f_var: 最终方差
        """
        # 计算平均方差（所有预测轨迹在所有时间步的方差）
        a_var = pred_trajs[..., :end_frame, :].var(dim=(1, 3)).mean(dim=1).sum()
        # 计算最终方差（所有预测轨迹在最终时间步的方差）
        f_var = pred_trajs[..., end_frame - 1, :].var(dim=(1, 2)).sum()
        return a_var, f_var

    def compute_MASD(self, pred_trajs, end_frame):
        """计算 MASD（最大平均采样距离）
        
        用于评估预测轨迹的多样性，计算所有预测轨迹对之间的最大距离。
        
        Args:
            pred_trajs: 预测轨迹张量，形状为 [b*num_agents, k_preds, future_frames, dim]
            end_frame: 结束帧索引
        
        Returns:
            masd: 最大平均采样距离
        """
        # 重塑为 (B, T, N, D) 形状以进行成对计算
        predictions = pred_trajs[:, :, :end_frame, :].permute(
            0, 2, 1, 3
        )  # Shape: (B, T, N, D)

        # 计算每个 (B, T) 位置的 N 个样本之间的成对 L2 距离
        pairwise_distances = torch.cdist(
            predictions, predictions, p=2
        )  # Shape: (B, T, N, N)

        # 获取所有对之间的最大距离（排除对角线）
        max_squared_distance = pairwise_distances.max(dim=-1)[0].max(dim=-1)[
            0
        ]  # Shape: (B, T)

        # 计算最终的 MASD 指标
        masd = max_squared_distance.mean(dim=-1).sum()
        return masd

    @torch.no_grad()
    def test(self, mode, eval_on_train=False, save_for_vis=False):
        """模型测试
        
        加载指定的 checkpoint 并在测试集（或训练集）上评估模型性能。
        
        Args:
            mode (str or int): 测试模式，可选值：
                - "last": 加载最后一个 checkpoint
                - "best": 加载最佳 checkpoint
                - 整数: 加载指定索引的最佳 checkpoint
            eval_on_train (bool, optional): 是否在训练集上评估，默认为 False
            save_for_vis (bool, optional): 是否保存可视化数据，默认为 False
        """
        # 初始化
        self.logger.info(f"testing start with the {mode} ckpt")
        self.mode = mode
        self.save_for_vis = save_for_vis

        # 设置随机种子
        set_random_seed(42)

        # 加载 checkpoint
        if mode == "last":
            ckpt_states = torch.load(
                os.path.join(self.cfg.model_dir, "checkpoint_last.pt"),
                map_location=self.device,
                weights_only=True,
            )
        elif mode == "best":
            ckpt_states = torch.load(
                os.path.join(self.cfg.model_dir, "checkpoint_best.pt"),
                map_location=self.device,
                weights_only=True,
            )
        elif isinstance(mode, int):
            ckpt_states = torch.load(
                os.path.join(self.cfg.model_dir, f"checkpoint_best_{mode}.pt"),
                map_location=self.device,
                weights_only=True,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        # 加载模型状态
        self.denoiser = self.accelerator.unwrap_model(self.denoiser)
        self.denoiser.load_state_dict(ckpt_states["model"])
        # 加载 EMA 状态
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(ckpt_states["ema"])

        # 评估模型
        if eval_on_train:
            fut_traj_gt, _, _ = self.eval_dataloader(training_err_check=True)
        else:
            fut_traj_gt, _, _ = self.eval_dataloader(testing_mode=True)
        self.logger.info(f"testing complete with the {mode} ckpt")

    def sample_from_denoising_model(self, data):
        """从去噪模型中获取采样结果
        
        从去噪模型中采样轨迹预测结果，并进行数据反归一化处理。
        
        Args:
            data: 输入数据字典
        
        Returns:
            pred_traj_y: 预测的未来轨迹，形状为 [(B*A), K, F, 2]
            pred_traj_x: 预测的过去轨迹，形状为 [(B*A), K, P, 2]
            pred_traj_y_at_t: 不同时间步的未来轨迹预测，形状为 [(B*A), T, K, F, 2]
            pred_traj_x_at_t: 不同时间步的过去轨迹预测，为 None
            t_seq: 时间步序列
            y_t_seq: 未来轨迹的噪声序列
            x_t_seq: 过去轨迹的噪声序列
            pred_score_y: 未来轨迹的预测分数
            pred_score_x: 过去轨迹的预测分数
        """
        # 从去噪模型中采样
        # 返回格式：y_t, y_data_at_t_ls, t_ls, y_t_ls, model_preds.pred_score
        # 形状：[B, K, A, T*F], [B, S, K, A, T*F], ,[B, S, K, A, T*F], [B, K, A]
        (
            pred_traj_y,
            pred_traj_y_at_t,
            y_t_seq,
            pred_traj_x,
            pred_traj_x_at_t,
            x_t_seq,
            t_seq,
            pred_score_y,
            pred_score_x,
        ) = self.denoiser.sample(
            data,
            num_trajs=self.cfg.denoising_head_preds,
            return_all_states=self.save_samples,
        )
        
        # 验证预测轨迹的形状
        assert pred_traj_y.shape[3] == self.cfg.MODEL.MODEL_OUT_DIM
        assert pred_traj_x.shape[3] == self.cfg.MODEL.MODEL_IN_DIM

        # 重塑未来轨迹：[B, k_preds, A, F*2] -> [B*A, k_preds, F, 2]
        pred_traj_y = rearrange(
            pred_traj_y, "b k a (f d) -> (b a) k f d", f=self.cfg.future_frames
        )[
            ..., 0:2
        ]  # [B, k_preds, 11, 24] -> [B * 11, k_preds, 12, 2]
        
        # 重塑过去轨迹：[B, k_preds, A, P*2] -> [B*A, k_preds, P, 2]
        pred_traj_x = rearrange(
            pred_traj_x, "b k a (p d) -> (b a) k p d", p=self.cfg.past_frames
        )[
            ..., 0:2
        ]  # [B, k_preds, 11, 16] -> [B * 11, k_preds, 8, 2]

        # 重塑不同时间步的未来轨迹
        pred_traj_y_at_t = rearrange(
            pred_traj_y_at_t, "b t k a (f d) -> (b a) t k f d", f=self.cfg.future_frames
        )[
            ..., 0:2
        ]  # [B, k_preds, 11, 24] -> [B * 11, k_preds, 12, 2]
        
        # 数据反归一化
        if self.cfg.get("data_norm", None) == "min_max":
            pred_traj_y = unnormalize_min_max(
                pred_traj_y, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1
            )
            pred_traj_y_at_t = unnormalize_min_max(
                pred_traj_y_at_t, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1
            )
            pred_traj_x = unnormalize_min_max(
                pred_traj_x, self.cfg.past_traj_min, self.cfg.past_traj_max, -1, 1
            )
            pred_traj_x_at_t = None
        else:
            raise NotImplementedError(
                f"Data normalization [{self.cfg.data_norm}] is not implemented yet."
            )

        return (
            pred_traj_y,
            pred_traj_x,
            pred_traj_y_at_t,
            pred_traj_x_at_t,
            t_seq,
            y_t_seq,
            x_t_seq,
            pred_score_y,
            pred_score_x,
        )

    def save_latent_states(
        self, t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls, file_name
    ):
        """保存去噪样本的潜在状态
        
        保存去噪过程中的中间状态，包括时间步、噪声序列、预测数据和预测分数。
        
        Args:
            t_seq_ls: 时间步序列列表
            y_t_seq_ls: 未来轨迹的噪声序列列表
            y_pred_data_ls: 未来轨迹的预测数据列表
            x_data_ls: 输入数据列表
            pred_score_ls: 预测分数列表
            file_name: 保存文件名
        """
        self.logger.info("Begin to save the denoising samples...")

        # 根据数据集类型确定需要保存的键
        if self.cfg.dataset in ["tbd", "t2fpv", "nba", "sdd", "eth_ucy"]:
            keys_to_save = [
                "past_traj",
                "fut_traj",
                "past_traj_original_scale",
                "fut_traj_original_scale",
                "fut_traj_vel",
            ]
        else:
            raise NotImplementedError(
                f"Dataset [{self.cfg.dataset}] is not implemented yet."
            )

        # 初始化保存字典
        states_to_save = {k: [] for k in keys_to_save}
        states_to_save["t"] = []
        states_to_save["y_t"] = []
        states_to_save["y_pred_data"] = []
        states_to_save["pred_score"] = []

        # 收集数据
        for i_batch, (t_seq, y_t_seq, y_pred_data, x_data, pred_score) in enumerate(
            zip(t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls, pred_score_ls)
        ):
            # 保存时间步
            t = t_seq.detach().cpu().numpy().reshape(1, -1)
            states_to_save["t"].append(t)

            # 保存未来轨迹的噪声序列
            y_t_seq = y_t_seq.detach().cpu().numpy()
            states_to_save["y_t"].append(y_t_seq)

            # 保存未来轨迹的预测数据
            y_pred_data = y_pred_data.detach().cpu().numpy()
            states_to_save["y_pred_data"].append(y_pred_data)

            # 保存预测分数
            pred_score = pred_score.detach().cpu().numpy()
            states_to_save["pred_score"].append(pred_score)

            # 保存输入数据中的相关键
            for key in keys_to_save:
                x_data_val_ = x_data[key].detach().cpu().numpy()
                assert len(y_t_seq) == len(x_data_val_)
                states_to_save[key].append(x_data_val_)

        # 合并数据
        for key in states_to_save:
            states_to_save[key] = np.concatenate(states_to_save[key], axis=0)

        # 清理配置，移除路径相关字段
        cfg_ = copy.deepcopy(self.cfg.yml_dict)

        def _remove_path_fields(cfg):
            """递归移除配置中的路径相关字段"""
            for k in list(cfg.keys()):
                if "path" in k or "dir" in k:
                    cfg.pop(k)
                elif isinstance(cfg[k], dict):
                    _remove_path_fields(cfg[k])
                else:
                    try:
                        if os.path.isdir(cfg[k]) or os.path.isfile(cfg[k]):
                            cfg.pop(k)
                    except:
                        pass

        _remove_path_fields(cfg_)

        # 添加元数据
        num_datapoints = len(states_to_save["y_t"])
        meta_data = {"cfg": cfg_, "size": num_datapoints}
        states_to_save["meta_data"] = meta_data

        # 保存数据到 pickle 文件
        save_path = os.path.join(self.cfg.sample_dir, f"{file_name}.pkl")
        self.logger.info("Saving the denoising samples to {}".format(save_path))
        pickle.dump(states_to_save, open(save_path, "wb"))

    def compute_pearson_corr(self, ade, score):
        """计算 ADE 和预测分数之间的皮尔逊相关系数
        
        用于评估预测分数与实际预测误差之间的相关性。
        
        Args:
            ade: 平均位移误差张量
            score: 预测分数张量
        
        Returns:
            corr: 皮尔逊相关系数
        """
        # 展平张量
        ade = ade.flatten()
        score = score.flatten()
        # 堆叠张量并计算相关系数
        combi = torch.stack([ade, score], dim=0)
        return torch.corrcoef(combi)[0, 1]

    def compute_k_agent_from_distance(
        self, gt_traj, pred_traj, agent_mask, k, pred_score
    ):
        """计算前 k 个预测的 ADE/FDE（基于每个智能体）
        
        计算前 k 个预测轨迹的 ADE（平均位移误差）和 FDE（最终位移误差），基于每个智能体。
        
        Args:
            gt_traj: 真实轨迹，形状为 [B*A, K, T, D]
            pred_traj: 预测轨迹，形状为 [B*A, K, T, D]
            agent_mask: 智能体掩码，形状为 [B, A]
            k: 前 k 个预测
            pred_score: 预测分数，形状为 [B, K, A]
        
        Returns:
            ade_best: 前 k 个预测中最佳轨迹的 ADE 之和
            fde_best: 前 k 个预测中最佳轨迹的 FDE 之和
            ade_avg: 前 k 个预测的平均 ADE 之和
            fde_avg: 前 k 个预测的平均 FDE 之和
        """
        # 提取有效的轨迹
        valid_idx = agent_mask.bool().view(-1)
        gt_traj = gt_traj[valid_idx]
        pred_traj = pred_traj[valid_idx]
        pred_score = rearrange(pred_score, "b k a -> (b a) k").unsqueeze(-1)
        pred_score = pred_score[valid_idx]

        # 计算预测轨迹与真实轨迹之间的距离
        distances = (gt_traj - pred_traj).norm(p=2, dim=-1)  # [valid_idx, K, T]
        num_traj = distances.shape[0]

        # 选择前 k 个预测的方式
        metrics_k_mode = self.cfg.get("metrics_k_mode", "min")
        if metrics_k_mode == "min":
            # 选择分数最高的 k 个预测
            _, selected_k = pred_score.topk(k, dim=1)
            selected_k = selected_k.repeat(1, 1, distances.shape[2])
            distances_k = distances.gather(1, selected_k)
        elif metrics_k_mode == "randn":
            # 随机选择 k 个预测
            selected_k = torch.randint(
                0,
                self.cfg.denoising_head_preds,
                (num_traj, k, distances.shape[2]),
                device=distances.device,
            )
            distances_k = distances.gather(1, selected_k)
        else:
            raise ValueError(f"Unknown metrics_k_mode: {metrics_k_mode}")

        # 计算 ADE 和 FDE
        # 对时间步平均，对 k 个预测取最小值，对有效智能体求和
        ade_best = distances_k.mean(dim=-1).min(dim=-1).values.sum(dim=0)
        fde_best = distances_k[..., -1].min(dim=-1).values.sum(dim=0)
        ade_avg = distances_k.mean(dim=-1).mean(dim=-1).sum(dim=0)
        fde_avg = distances_k[..., -1].mean(dim=-1).sum(dim=0)

        return ade_best, fde_best, ade_avg, fde_avg

    def compute_k_scene_from_distance(
        self, gt_traj, pred_traj, agent_mask, k, pred_score
    ):
        """计算前 k 个预测的 JADE/JFDE（基于场景）
        
        计算前 k 个预测轨迹的 JADE（联合平均位移误差）和 JFDE（联合最终位移误差），基于场景。
        
        Args:
            gt_traj: 真实轨迹，形状为 [B*A, K, T, D]
            pred_traj: 预测轨迹，形状为 [B*A, K, T, D]
            agent_mask: 智能体掩码，形状为 [B, A]
            k: 前 k 个预测
            pred_score: 预测分数，形状为 [B, K, A]
        
        Returns:
            jade_best: 前 k 个预测中最佳轨迹的 JADE 之和
            jfde_best: 前 k 个预测中最佳轨迹的 JFDE 之和
            jade_avg: 前 k 个预测的平均 JADE 之和
            jfde_avg: 前 k 个预测的平均 JFDE 之和
        """
        B, A = agent_mask.shape

        # 重塑轨迹为场景维度
        gt_traj = rearrange(gt_traj, "(b a) k t d -> b a k t d", b=B, a=A)
        pred_traj = rearrange(pred_traj, "(b a) k t d -> b a k t d", b=B, a=A)
        # 计算预测轨迹与真实轨迹之间的距离
        distances = (gt_traj - pred_traj).norm(p=2, dim=-1)  # [B, A, K, T]

        # 选择前 k 个预测的方式
        metrics_k_mode = self.cfg.get("metrics_k_mode", "min")
        pred_score = rearrange(pred_score, "b k a -> b a k")
        if metrics_k_mode == "min":
            # 选择分数最高的 k 个预测
            _, selected_k = pred_score.topk(k, dim=-1)  # [B, A, k]
            selected_k = repeat(selected_k, "b a k ->  b a k t", t=distances.shape[-1])
            distances_k = distances.gather(2, selected_k)  # [B, A, k, T]
        elif metrics_k_mode == "randn":
            # 随机选择 k 个预测
            selected_k = torch.randint(
                0,
                self.cfg.denoising_head_preds,
                (B, A, k, distances.shape[-1]),
                device=distances.device,
            )
            distances_k = distances.gather(2, selected_k)
        else:
            raise ValueError(f"Unknown metrics_k_mode: {metrics_k_mode}")

        # 处理智能体掩码
        agent_mask_ = agent_mask
        agent_mask = repeat(agent_mask, "b a -> b a k t", k=k, t=distances.shape[-1])
        assert agent_mask_.sum(dim=1).min() > 0

        # 计算 JADE（联合平均位移误差）
        jade = (distances_k * agent_mask).mean(dim=-1).sum(dim=1) / agent_mask_.sum(
            dim=1
        ).unsqueeze(-1)
        jade_best = jade.sum(dim=0).min(dim=-1).values
        jade_avg = jade.sum(dim=0).mean(dim=-1)

        # 计算 JFDE（联合最终位移误差）
        jfde = (distances_k[..., -1] * agent_mask[..., -1]).sum(
            dim=1
        ) / agent_mask_.sum(dim=1).unsqueeze(-1)
        jfde_best = jfde.sum(dim=0).min(dim=-1).values
        jfde_avg = jfde.sum(dim=0).mean(dim=-1)

        return jade_best, jfde_best, jade_avg, jfde_avg

    def compute_k_metrics(
        self, gt_traj, pred_traj, agent_mask, pred_score, performance, performance_joint
    ):
        """计算不同 k 值的评估指标
        
        计算不同 k 值（前 k 个预测）的 ADE、FDE、JADE 和 JFDE 指标。
        
        Args:
            gt_traj: 真实轨迹，形状为 [B, A, T, D]
            pred_traj: 预测轨迹，形状为 [(B*A), K, T, D]
            agent_mask: 智能体掩码，形状为 [B, A]
            pred_score: 预测分数，形状为 [B, K, A]
            performance: 性能字典，存储 ADE 和 FDE 指标
            performance_joint: 联合性能字典，存储 JADE 和 JFDE 指标
        
        Returns:
            performance: 更新后的性能字典
            performance_joint: 更新后的联合性能字典
        """
        # 重塑真实轨迹为 [(B*A), T, D]，并添加预测维度
        gt_traj = rearrange(gt_traj, "b a t d -> (b a) t d")
        gt_traj = gt_traj.unsqueeze(1).repeat(1, self.cfg.denoising_head_preds, 1, 1)

        # 获取 k 值列表
        k_list = self.cfg.get("K_LIST", [1, 3, 5, 20])
        # 对每个 k 值计算指标
        for idx, k in enumerate(k_list):
            # 计算基于智能体的 ADE 和 FDE
            ade, fde, ade_avg, fde_avg = self.compute_k_agent_from_distance(
                gt_traj, pred_traj, agent_mask, k, pred_score
            )
            performance["ADE_min"][idx] += ade.item()
            performance["FDE_min"][idx] += fde.item()
            performance["ADE_avg"][idx] += ade_avg.item()
            performance["FDE_avg"][idx] += fde_avg.item()

            # 计算基于场景的 JADE 和 JFDE
            jade, jfde, jade_avg, jfde_avg = self.compute_k_scene_from_distance(
                gt_traj, pred_traj, agent_mask, k, pred_score
            )

            performance_joint["JADE_min"][idx] += jade.item()
            performance_joint["JFDE_min"][idx] += jfde.item()
            performance_joint["JADE_avg"][idx] += jade_avg.item()
            performance_joint["JFDE_avg"][idx] += jfde_avg.item()

        return performance, performance_joint

    def eval_dataloader(self, testing_mode=False, training_err_check=False):
        """评估数据加载器/数据集
        
        在指定的数据集上评估模型性能，计算各种评估指标，包括 ADE、FDE、JADE 和 JFDE。
        
        Args:
            testing_mode (bool, optional): 是否在测试集上评估，默认为 False
            training_err_check (bool, optional): 是否在训练集上评估，默认为 False
        
        Returns:
            fut_traj_gt: 真实未来轨迹，形状为 [(B*A), T, D]
            performance_future: 未来轨迹的性能指标字典
            num_trajs: 有效轨迹数量
        """
        # 开启评估模式
        self.denoiser.eval()
        self.ema.ema_model.eval()
        self.logger.info(f"Record the statistics of samples from the denoising model")

        # 选择评估数据集
        if testing_mode:
            self.logger.info(f"Start recording test set ADE/FDE...")
            status = "test"
            dl = self.test_loader
        elif training_err_check:
            self.logger.info(f"Start recording training set ADE/FDE...")
            status = "train"
            dl = self.train_loader
        else:
            self.logger.info(f"Start recording validation set ADE/FDE...")
            status = "val"
            dl = self.val_loader

        # 初始化性能字典
        performance_future = {
            "FDE_min": [0, 0, 0, 0],
            "ADE_min": [0, 0, 0, 0],
            "FDE_avg": [0, 0, 0, 0],
            "ADE_avg": [0, 0, 0, 0],
        }
        performance_joint_future = {
            "JFDE_min": [0, 0, 0, 0],
            "JADE_min": [0, 0, 0, 0],
            "JFDE_avg": [0, 0, 0, 0],
            "JADE_avg": [0, 0, 0, 0],
        }
        performance_past = {
            "ADE_min": [0, 0, 0, 0],
            "FDE_min": [0, 0, 0, 0],
            "ADE_avg": [0, 0, 0, 0],
            "FDE_avg": [0, 0, 0, 0],
        }
        performance_joint_past = {
            "JFDE_min": [0, 0, 0, 0],
            "JADE_min": [0, 0, 0, 0],
            "JFDE_avg": [0, 0, 0, 0],
            "JADE_avg": [0, 0, 0, 0],
        }
        num_trajs = 0
        num_scenes = 0

        # 初始化保存列表
        t_seq_ls, y_t_seq_ls, y_pred_data_ls, x_data_ls = [], [], [], []
        
        # 记录运行时间
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        # 遍历数据集
        for i_batch, data in enumerate(dl):
            bs = int(data["batch_size"])
            data = {k: v.to(self.device) for k, v in data.items()}

            # 从去噪模型中采样
            (
                pred_traj_y,
                pred_traj_x,
                pred_traj_y_at_t,
                pred_traj_x_at_t,
                t_seq,
                y_t_seq,
                x_t_seq,
                pred_score_y,
                pred_score_x,
            ) = self.sample_from_denoising_model(
                data
            )  # pred_traj: [B*A, K, F, 2], pred_traj_t: [B*A, T, K, F, 2]

            # 获取智能体掩码
            if self.cfg.get("use_ablation_dataset", False):
                B, A = data["past_traj"].shape[:2]
                agent_mask = torch.ones((B, A), device=pred_traj_y.device)
            else:
                agent_mask = data["agent_mask"]

            # 计算未来轨迹的评估指标
            performance_future, performance_joint_future = self.compute_k_metrics(
                data["fut_traj_original_scale"],
                pred_traj_y,
                agent_mask,
                pred_score_y,
                performance_future,
                performance_joint_future,
            )
            # 计算过去轨迹的评估指标
            performance_past, performance_joint_past = self.compute_k_metrics(
                data["past_traj_gt_original_scale"][..., :2],
                pred_traj_x,
                agent_mask,
                pred_score_x,
                performance_past,
                performance_joint_past,
            )

            # 更新有效轨迹和场景数量
            num_trajs += agent_mask.sum().item()  # 有效智能体数量
            num_scenes += bs  # 批次大小 B

            # 保存去噪样本（用于 IMLE）
            if self.save_samples:
                raise NotImplementedError("Not implemented yet")

            # 保存可视化样本
            if testing_mode and self.save_for_vis:
                save_tensors = dict()
                save_tensors["hist_obs"] = data["past_traj_original_scale"][
                    ..., :2
                ]  # [B, A, F, 2]
                save_tensors["hist_gt"] = data["past_traj_gt_original_scale"][
                    ..., :2
                ]  # [B, A, F, 2]
                save_tensors["fut_gt"] = data["fut_traj_original_scale"]  # [B, A, T, 2]

                save_tensors["fut_pred"] = pred_traj_y  # [B*A, K, F, 2]
                save_tensors["fut_pred_at_t"] = pred_traj_y_at_t
                save_tensors["hist_pred"] = pred_traj_x
                save_tensors["agent_mask"] = agent_mask
                save_tensors["past_theta"] = data["past_theta"]

                self.save_batch_samples(save_tensors, i_batch)

        # 结束时间记录
        end.record()
        torch.cuda.synchronize()
        self.logger.info(f"Total runtime: {start.elapsed_time(end):5f} ms")
        self.logger.info(
            f"Runtime per scene: {start.elapsed_time(end)/len(dl.dataset):5f} ms"
        )
        self.logger.info(f"Number of scenes: {dl.dataset}")
        
        # 计算当前 epoch
        steps_per_epoch = self.train_num_steps // max(
            self.cfg.OPTIMIZATION.NUM_EPOCHS, 1
        )
        cur_epoch = self.step // steps_per_epoch if steps_per_epoch > 0 else 0

        # 获取 k 值列表
        k_list = self.cfg.get("K_LIST", [1, 3, 5, 20])

        # 记录到 TensorBoard（非测试模式）
        if not testing_mode:
            self.logger.info(
                f"{self.step}/{self.train_num_steps}, running inference on {num_trajs} agents (trajectories)"
            )
            for idx, k in enumerate(k_list):
                if self.tb_log:
                    # 未来轨迹指标
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_ADE_min_k{k}",
                        performance_future["ADE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_FDE_min_k{k}",
                        performance_future["FDE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_ADE_avg_k{k}",
                        performance_future["ADE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/F_FDE_avg_k{k}",
                        performance_future["FDE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JADE_min_k{k}",
                        performance_joint_future["JADE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JFDE_min_k{k}",
                        performance_joint_future["JFDE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JADE_avg_k{k}",
                        performance_joint_future["JADE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/J_JFDE_avg_k{k}",
                        performance_joint_future["JFDE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )

                    # 过去轨迹指标
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_ADE_min_k{k}",
                        performance_past["ADE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_FDE_min_k{k}",
                        performance_past["FDE_min"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_ADE_avg_k{k}",
                        performance_past["ADE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_FDE_avg_k{k}",
                        performance_past["FDE_avg"][idx] / num_trajs,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JADE_min_k{k}",
                        performance_joint_past["JADE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JFDE_min_k{k}",
                        performance_joint_past["JFDE_min"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JADE_avg_k{k}",
                        performance_joint_past["JADE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )
                    self.tb_log.add_scalar(
                        f"eval_{status}/P_JFDE_avg_k{k}",
                        performance_joint_past["JFDE_avg"][idx] / num_scenes,
                        cur_epoch,
                    )

        # 打印性能指标
        # 未来轨迹指标
        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--ADE_min(k{}): {:.7f}\t--FDE_min(k{}): {:.7f}".format(
                    k,
                    performance_future["ADE_min"][idx] / num_trajs,
                    k,
                    performance_future["FDE_min"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--ADE_avg(k{}): {:.7f}\t--FDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_future["ADE_avg"][idx] / num_trajs,
                    k,
                    performance_future["FDE_avg"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--JADE_min(k{}): {:.7f}\t--JFDE_min(k{}): {:.7f}".format(
                    k,
                    performance_joint_future["JADE_min"][idx] / num_scenes,
                    k,
                    performance_joint_future["JFDE_min"][idx] / num_scenes,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Future--JADE_avg(k{}): {:.7f}\t--JFDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_joint_future["JADE_avg"][idx] / num_scenes,
                    k,
                    performance_joint_future["JFDE_avg"][idx] / num_scenes,
                )
            )

        # 过去轨迹指标
        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--ADE_min(k{}): {:.7f}\t--FDE_min(k{}): {:.7f}".format(
                    k,
                    performance_past["ADE_min"][idx] / num_trajs,
                    k,
                    performance_past["FDE_min"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--ADE_avg(k{}): {:.7f}\t--FDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_past["ADE_avg"][idx] / num_trajs,
                    k,
                    performance_past["FDE_avg"][idx] / num_trajs,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--JADE_min(k{}): {:.7f}\t--JFDE_min(k{}): {:.7f}".format(
                    k,
                    performance_joint_past["JADE_min"][idx] / num_scenes,
                    k,
                    performance_joint_past["JFDE_min"][idx] / num_scenes,
                )
            )

        for idx, k in enumerate(k_list):
            self.logger.info(
                "--Past--JADE_avg(k{}): {:.7f}\t--JFDE_avg(k{}): {:.7f}".format(
                    k,
                    performance_joint_past["JADE_avg"][idx] / num_scenes,
                    k,
                    performance_joint_past["JFDE_avg"][idx] / num_scenes,
                )
            )

        # 重塑真实未来轨迹并返回
        fut_traj_gt = rearrange(data["fut_traj_original_scale"], "b a t d -> (b a) t d")
        return fut_traj_gt, performance_future, num_trajs

    def save_batch_samples(self, save_tensors, i_batch, prefix="visual"):
        """保存批次样本
        
        保存批次样本用于可视化，包括真实轨迹、预测轨迹和相关信息。
        
        Args:
            save_tensors: 包含样本数据的字典
            i_batch: 批次索引
            prefix: 保存文件的前缀，默认为 "visual"
        """
        # 构建保存路径
        save_path = os.path.join(
            self.cfg.sample_dir, f"{prefix}_mode_{self.mode}_batch_{i_batch}.pkl"
        )

        # 重塑未来轨迹
        fut_gt = rearrange(save_tensors["fut_gt"], "b a f d -> (b a) f d")
        fut_gt_ = repeat(fut_gt, "b f d -> b k f d", k=self.cfg.denoising_head_preds)
        fut_pred = save_tensors["fut_pred"]
        fut_pred_at_t = save_tensors["fut_pred_at_t"]  # [B*A, K, T_denoising, F, 2]

        # 重塑过去轨迹
        hist_gt = rearrange(save_tensors["hist_gt"], "b a f d -> (b a) f d")
        hist_gt_ = repeat(hist_gt, "b f d -> b k f d", k=self.cfg.denoising_head_preds)
        hist_pred = save_tensors["hist_pred"]
        hist_obs = rearrange(save_tensors["hist_obs"], "b a f d -> (b a) f d")

        # 提取有效样本
        valid_idx = save_tensors["agent_mask"].bool().flatten()
        past_theta = save_tensors["past_theta"].flatten()[valid_idx]
        fut_gt = fut_gt[valid_idx]
        fut_pred = fut_pred[valid_idx]
        fut_pred_at_t = fut_pred_at_t[valid_idx]
        hist_obs = hist_obs[valid_idx]
        hist_gt = hist_gt[valid_idx]
        hist_pred = hist_pred[valid_idx]

        # 更新保存张量
        save_tensors["fut_gt"] = fut_gt
        save_tensors["fut_pred"] = fut_pred
        save_tensors["fut_pred_at_t"] = fut_pred_at_t
        save_tensors["hist_obs"] = hist_obs
        save_tensors["hist_gt"] = hist_gt
        save_tensors["hist_pred"] = hist_pred
        save_tensors["past_theta"] = past_theta

        # 重塑用于计算距离的张量
        fut_gt_ = fut_gt_[valid_idx]
        hist_gt_ = hist_gt_[valid_idx]

        # 计算最佳预测轨迹的索引
        distances_fut = (fut_gt_ - fut_pred).norm(p=2, dim=-1).sum(dim=-1)
        save_tensors["fut_best_idx"] = distances_fut.argmin(dim=-1)

        distances_hist = (hist_gt_ - hist_pred).norm(p=2, dim=-1).sum(dim=-1)
        save_tensors["hist_best_idx"] = distances_hist.argmin(dim=-1)

        # 将张量转换为 numpy 数组
        for k, v in save_tensors.items():
            save_tensors[k] = v.detach().cpu().numpy()

        # 保存到 pickle 文件
        pickle.dump(save_tensors, open(save_path, "wb"))
        return
