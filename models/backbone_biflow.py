"""BiFlow 模型骨干网络模块

本模块实现了 BiFlow 模型的骨干网络，包括上下文编码器、特征融合器和运动解码器等组件。
"""

import numpy as np

import torch
import torch.nn as nn
from .context_encoder import build_context_encoder
from .motion_decoder import build_decoder
from .feature_fuser import build_feature_fuser
from .utils.common_layers import build_mlps
from einops import repeat, rearrange
from models.utils.contextual_scorer import AnchorHead


class BiFlowModel(nn.Module):
    """BiFlow 模型类
    
    实现了双向流匹配模型，包含上下文编码器、特征融合器、未来运动解码器和过去运动解码器。
    """
    
    def __init__(self, model_config, logger, config):
        """
        初始化 BiFlowModel
        
        Args:
            model_config: 模型配置
            logger: 日志记录器
            config: 全局配置
        """
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.config = config
        self.past_frames = self.config.get("past_frames", 8)

        use_pre_norm = self.model_cfg.get("USE_PRE_NORM", False)
        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.use_mask = self.model_cfg.get("USE_MASK", False)
        self.use_imputation = self.model_cfg.get("USE_IMPUTATION", False)

        self.use_anchor = self.model_cfg.get("USE_ANCHOR", False)
        self.use_hist_cond = self.model_cfg.get("USE_HIST_COND", False)

        # 初始化锚点头（如果启用）
        if self.use_anchor:
            self.anchor_head = AnchorHead(self.dim)

        # 构建上下文编码器
        self.context_encoder = build_context_encoder(
            self.model_cfg.CONTEXT_ENCODER, use_pre_norm
        )

        # 构建特征融合器
        self.feature_fuser = build_feature_fuser(self.config)

        # 构建未来运动解码器
        self.motion_decoder_fut = build_decoder(
            self.model_cfg.MOTION_DECODER, use_pre_norm, use_anchor=self.use_anchor
        )

        # 构建过去运动解码器
        self.motion_decoder_pst = build_decoder(
            self.model_cfg.MOTION_DECODER, use_pre_norm
        )

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        # 构建未来轨迹回归头
        self.reg_head_fut = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.REGRESSION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )
        # 构建未来轨迹分类头
        self.cls_head_fut = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.CLASSIFICATION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )

        # 构建过去轨迹回归头
        self.reg_head_pst = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.RECONSTRUCTION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )
        # 构建过去轨迹分类头
        self.cls_head_pst = build_mlps(
            c_in=dim_decoder,
            mlp_channels=self.model_cfg.CLASSIFICATION_MLPS,
            ret_before_act=True,
            without_norm=True,
        )

        # 检查是否冻结过去分支
        pst_frozen = self.config.OPTIMIZATION.LOSS_WEIGHTS["branch_past"] < 1e-6
        if pst_frozen:
            # 冻结过去解码器的参数
            for p in self.motion_decoder_pst.parameters():
                p.requires_grad = False
            for p in self.cls_head_pst.parameters():
                p.requires_grad = False
            for p in self.reg_head_pst.parameters():
                p.requires_grad = False

        # 打印模型参数数量
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_fuser = sum(p.numel() for p in self.feature_fuser.parameters())
        params_decoder_fut = sum(
            p.numel() for p in self.motion_decoder_fut.parameters()
        )
        params_decoder_pst = sum(
            p.numel() for p in self.motion_decoder_pst.parameters()
        )
        params_total = sum(p.numel() for p in self.parameters())
        params_other = (
            params_total
            - params_encoder
            - params_decoder_fut
            - params_decoder_pst
            - params_fuser
        )
        logger.info(
            "Total parameters: {:,}, Encoder: {:,}, Fuser: {:,}, Decoder_fut: {:,}, Decoder_pst: {:,}, Other: {:,}".format(
                params_total,
                params_encoder,
                params_fuser,
                params_decoder_fut,
                params_decoder_pst,
                params_other,
            )
        )

    def forward(self, y_t_in, t_fut, x_t_in, t_pst, x_data):
        """
        前向传播函数
        
        Args:
            y_t_in: 噪声向量（未来轨迹）
            t_fut: 未来轨迹的去噪时间步
            x_t_in: 噪声向量（过去轨迹）
            t_pst: 过去轨迹的去噪时间步
            x_data: 数据字典，包含以下键：
                - past_traj: 过去轨迹
                - future_traj: 未来轨迹
                - future_traj_vel: 未来轨迹速度
                - trajectory mask: 轨迹掩码（可能存在）
                - batch_size: 批次大小
                - indexes: 当执行 IMLE 时存在
                
        Returns:
            tuple: (过去轨迹去噪结果, 过去轨迹分类分数, 未来轨迹去噪结果, 未来轨迹分类分数)
        """
        # 验证输入形状
        assert y_t_in.shape[-1] == 24, "y shape is not correct"
        assert x_t_in.shape[-1] == 16, "x shape is not correct"

        B, K, A, _ = y_t_in.shape

        # 获取历史条件
        past_traj = x_data["past_traj_original_scale"]  # [B, A, P, 6]
        past_traj_mask = x_data["past_traj_valid"]  # [B, A, P]
        # 根据是否使用缺失值插补处理过去轨迹
        past_traj = (
            past_traj
            if self.use_imputation
            else past_traj * past_traj_mask.unsqueeze(-1)
        )
        # 获取智能体掩码
        if self.config.get("use_ablation_dataset", False):
            agent_mask = torch.ones((B, A), device=past_traj.device)
        else:
            agent_mask = x_data["agent_mask"]  # [B, A]

        agent_score = None

        # 拼接有效掩码
        concat_past_traj_mask = (
            past_traj_mask.unsqueeze(-1)
            if self.use_mask
            else torch.zeros_like(past_traj_mask).unsqueeze(-1)
        )
        # TODO: feature encoder to handle frame_score
        concat_frame_score = torch.zeros_like(past_traj_mask).unsqueeze(
            -1
        )  # [B, A, P, 1]
        concat_list = [past_traj, concat_past_traj_mask, concat_frame_score]
        past_traj = torch.cat(concat_list, dim=-1)

        # 上下文编码
        encoder_out = self.context_encoder(past_traj, agent_mask, agent_score)
        # 扩展编码输出以匹配批次大小
        encoder_out_batch = repeat(encoder_out, "b a d -> b k a d", k=K)

        # 计算锚点（如果启用）
        anchor_agent, anchor_scene = None, None
        if self.use_anchor:
            anchor_agent, anchor_scene = self.anchor_head(
                encoder_out, agent_mask, agent_score
            )

        # 初始化嵌入
        if not self.use_hist_cond:
            x_cond = torch.zeros_like(encoder_out_batch)
        else:
            x_cond = encoder_out_batch
        # 特征融合
        y_query_token, x_query_token, y_t_emb, x_t_emb = self.feature_fuser(
            y_t_in,
            t_fut,
            encoder_out_batch,
            x_t_in,
            t_pst,
            x_cond,
            agent_mask,
            anchor_agent,
            anchor_scene,
        )

        # 解码未来和过去轨迹
        readout_token_fut = self.motion_decoder_fut(
            y_query_token, y_t_emb, agent_mask, anchor_agent, anchor_scene
        )
        readout_token_pst = self.motion_decoder_pst(x_query_token, x_t_emb, agent_mask)

        # 输出层
        denoiser_y = self.reg_head_fut(readout_token_fut)  # [B, K, A, F * D]
        denoiser_cls_fut = self.cls_head_fut(readout_token_fut).squeeze(-1)  # [B, K, A]
        denoiser_x = self.reg_head_pst(readout_token_pst)  # [B, K, A, P * D]
        denoiser_cls_pst = self.cls_head_pst(readout_token_pst).squeeze(-1)  # [B, K, A]

        return denoiser_x, denoiser_cls_pst, denoiser_y, denoiser_cls_fut
