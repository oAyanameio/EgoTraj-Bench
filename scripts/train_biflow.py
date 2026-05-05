"""BiFlow 模型训练脚本

该脚本用于训练 BiFlow 轨迹预测模型，包括参数解析、配置初始化、数据加载器构建、
网络构建和训练过程等功能。
"""

import copy
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import torch
import argparse

from tensorboardX import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
print(f"Using root directory: {ROOT_DIR}")

from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file
from utils.dataset_config import FOLD_CONFIG

from models.flow_matching_biflow import BiFlowMatcher
from models.backbone_biflow import BiFlowModel
from trainer.biflow_trainer import BiFlowTrainer

from loaders.dataloader_egotraj import EgoTrajDataset, seq_collate_egotraj


def parse_config():
    """解析命令行参数并返回配置选项
    
    Returns:
        解析后的命令行参数
    """

    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument(
        "--cfg",
        default="cfg/biflow_k20.yml",
        type=str,
        help="配置文件路径",
    )
    parser.add_argument(
        "--exp",
        default="train",
        type=str,
        help="每次运行的实验描述，保存文件夹的名称。",
    )

    # 数据配置
    parser.add_argument(
        "--fold_name",
        default="tbd",
        type=str,
        help="实验的折叠名称。",
    )
    parser.add_argument(
        "--data_source",
        default="original",
        type=str,
        choices=["original", "original_bal"],
        help="数据源: 'original' 用于 EgoTraj-TBD, 'original_bal' 用于 T2FPV-ETH。",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="覆盖配置文件中的 epoch 数量。",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="覆盖配置文件中的批大小。",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据存储的目录。如果未指定，将根据 fold_name 自动设置。",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="要使用的 GPU 设备 ID。",
    )
    parser.add_argument(
        "--overfit",
        default=False,
        action="store_true",
        help="通过将测试集设置为与训练集相同的条目来过度拟合测试集。",
    )
    parser.add_argument(
        "--checkpt_freq",
        default=1,
        type=int,
        help="覆盖配置文件中的 checkpt_freq。",
    )
    parser.add_argument(
        "--max_num_ckpts",
        default=5,
        type=int,
        help="覆盖配置文件中的 max_num_ckpts。",
    )
    parser.add_argument(
        "--data_norm",
        default="min_max",
        choices=["min_max", "original"],
        help="数据的归一化方法。",
    )
    parser.add_argument(
        "--rotate",
        type=bool,
        default=True,
        help="是否旋转数据集中的轨迹",
    )
    parser.add_argument(
        "--rotate_time_frame",
        type=int,
        default=6,
        help="旋转轨迹的时间帧索引。",
    )
    # 恢复训练
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="是否从检查点恢复训练。",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="checkpoint_last",
        help="用于恢复训练的检查点名称。",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=150,
        help="恢复训练的起始 epoch。",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=-1,
        help="训练的早停。-1 表示没有早停。",
    )

    # 可复现性配置
    parser.add_argument(
        "--fix_random_seed",
        type=bool,
        default=True,
        help="固定随机种子以确保可复现性",
    )
    parser.add_argument("--seed", type=int, default=9527, help="设置随机种子。")

    ### FM 参数 ###
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=10,
        help="FlowMatcher 的采样时间步数。",
    )

    # 训练期间的时间调度器
    parser.add_argument(
        "--t_schedule",
        type=str,
        choices=["uniform", "logit_normal"],
        default="logit_normal",
        help="FlowMatcher 的时间调度。",
    )
    parser.add_argument(
        "--logit_norm_mean",
        default=-0.5,
        type=float,
        help="logit 正态分布的均值。",
    )
    parser.add_argument(
        "--logit_norm_std",
        default=1.5,
        type=float,
        help="logit 正态分布的标准差。",
    )

    parser.add_argument(
        "--fm_wrapper",
        type=str,
        default="direct",
        choices=["direct", "velocity", "precond"],
        help="FlowMatcher 的包装器。",
    )
    parser.add_argument(
        "--fm_rew_sqrt",
        default=False,
        action="store_true",
        help="是否对重加权因子应用平方根。",
    )
    parser.add_argument(
        "--fm_in_scaling",
        type=bool,
        default=True,
        help="是否缩放 FlowMatcher 的输入。",
    )
    parser.add_argument(
        "--sigma_data",
        type=float,
        default=0.13,
        help="数据的标准差。",
    )
    # 输入 dropout / 掩码率
    parser.add_argument(
        "--drop_method",
        default="emb",
        type=str,
        choices=["None", "input", "emb"],
        help="FlowMatcher 的 dropout 方法。",
    )
    parser.add_argument(
        "--drop_logi_k",
        default=20.0,
        type=float,
        help="不同时间步的掩码率的逻辑增长率。",
    )
    parser.add_argument(
        "--drop_logi_m",
        default=0.5,
        type=float,
        help="不同时间步的掩码率的逻辑中点。",
    )
    ### FM 参数 ###

    ### 架构配置 ###
    parser.add_argument(
        "--use_pre_norm",
        default=False,
        action="store_true",
        help="是否在 Transformer 编码器中归一化输入轨迹。",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="覆盖配置文件中的层数。",
    )
    parser.add_argument(
        "--dropout",
        default=None,
        type=float,
        help="覆盖配置文件中的 dropout 率。",
    )
    ### 架构配置 ###

    ### 通用去噪目标配置 ###
    parser.add_argument(
        "--tied_noise",
        type=bool,
        default=True,
        help="是否为去噪器使用绑定噪声。",
    )
    ### 通用去噪目标配置 ###

    ### 回归损失配置 ###
    parser.add_argument(
        "--loss_nn_mode",
        type=str,
        default="agent",
        choices=["agent", "scene"],
        help="是否使用智能体级或场景级的 NN 损失。",
    )
    parser.add_argument(
        "--loss_reg_reduction",
        type=str,
        default="sum",
        choices=["mean", "sum"],
        help="回归损失的归约方法。",
    )
    ### 回归损失配置 ###

    ### 分类损失配置 ###
    parser.add_argument(
        "--loss_cls_weight",
        type=float,
        default=1.0,
        help="分类损失的权重。",
    )
    ### 分类损失配置 ###

    ### 优化配置 ###
    parser.add_argument(
        "--init_lr",
        type=float,
        default=1e-4,
        help="覆盖配置文件中的峰值学习率。",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="覆盖配置文件中的权重衰减。",
    )
    ### 优化配置 ###

    ### 精炼器配置 ###
    parser.add_argument(
        "--module_layers",
        type=int,
        default=1,
        help="模块的层数。",
    )
    parser.add_argument(
        "--loss_branch_past_weight",
        type=float,
        default=0.5,
        help="分支过去损失的权重。",
    )
    parser.add_argument(
        "--approx_t_std",
        type=float,
        default=5e-2,
        help="过去分支中近似时间的标准差。",
    )
    parser.add_argument(
        "--use_mask",
        type=bool,
        default=True,
        help="是否使用掩码。",
    )
    parser.add_argument(
        "--use_imputation",
        type=bool,
        default=True,
        help="是否使用插补。",
    )

    ### 精炼器配置 ###

    ### 锚点配置 ###
    parser.add_argument(
        "--use_anchor",
        type=bool,
        default=True,
        help="是否使用锚点。",
    )
    parser.add_argument(
        "--use_hist_cond",
        type=bool,
        default=True,
        help="是否使用历史条件。",
    )
    ### 锚点配置 ###

    ### 融合器配置 ###
    parser.add_argument(
        "--fuser_name",
        type=str,
        default="SharedFuser",
        choices=["SharedFuser"],
        help="要使用的融合器模块。",
    )
    parser.add_argument(
        "--max_num_agents",
        type=int,
        default=None,
        help="每个场景的最大智能体数量。如果未指定，将根据 fold_name 自动设置。",
    )
    ### 融合器配置 ###

    return parser.parse_args()


def init_basics(args, tag_prefix=None):
    """初始化实验的基本配置
    
    Args:
        args: 命令行参数
        tag_prefix: 标签前缀
    
    Returns:
        cfg: 配置对象
        logger: 日志记录器
        tb_log: TensorBoard 日志记录器
    """

    """加载配置文件"""
    cfg = Config(args.cfg, f"{args.exp}")

    # 保留配置中的 K 列表（发布路径只保留 K=20 配置）。
    cfg.K_LIST = cfg.get("K_LIST", [1, 3, 5, 20])
    cfg.USE_CLEAN_HIST = bool(tag_prefix is not None and "cln" in tag_prefix)

    # 顺序配置覆盖（扁平化，无嵌套更新助手）。
    if args.data_source not in {"original", "original_bal"}:
        raise ValueError(f"Invalid data source: {args.data_source}")

    # 将 args 中的 fold_name 同步到 cfg 中，以便日志反映真实的数据折叠。
    # 否则，当通过 --fold_name 传递 hotel/zara1/zara2 时，cfg.fold_name 会保持 yaml 默认值 ('eth')，导致日志条目误导。
    cfg.fold_name = args.fold_name

    # 更新模型配置
    cfg.MODEL.USE_ANCHOR = args.use_anchor
    cfg.MODEL.USE_HIST_COND = args.use_hist_cond
    cfg.OPTIMIZATION.LOSS_WEIGHTS["branch_past"] = args.loss_branch_past_weight
    cfg.MODEL.USE_MASK = args.use_mask
    cfg.MODEL.USE_IMPUTATION = args.use_imputation
    cfg.approx_t_std = args.approx_t_std

    # 检查融合器名称
    if args.fuser_name != "SharedFuser":
        raise NotImplementedError(f"Fuser [{args.fuser_name}] is not implemented yet.")
    cfg.MODEL.FUSER_NAME = args.fuser_name

    # 配置 Flow Matching 参数
    if cfg.denoising_method == "fm":
        cfg.sigma_data = args.sigma_data
        cfg.sampling_steps = args.sampling_steps
        cfg.t_schedule = args.t_schedule
        if args.t_schedule == "logit_normal":
            cfg.logit_norm_mean = args.logit_norm_mean
            cfg.logit_norm_std = args.logit_norm_std
        cfg.fm_wrapper = args.fm_wrapper
        cfg.fm_rew_sqrt = args.fm_rew_sqrt
        cfg.fm_in_scaling = args.fm_in_scaling
        if (
            args.drop_method is not None
            and args.drop_logi_k is not None
            and args.drop_logi_m is not None
        ):
            cfg.drop_method = args.drop_method
            cfg.drop_logi_k = args.drop_logi_k
            cfg.drop_logi_m = args.drop_logi_m

    # 更新模型架构配置
    cfg.MODEL.USE_PRE_NORM = args.use_pre_norm
    cfg.MODEL.NUM_LAYERS = args.num_layers
    cfg.MODEL.DROPOUT = args.dropout
    if args.num_layers is not None:
        cfg.MODEL.CONTEXT_ENCODER.NUM_ATTN_LAYERS = args.num_layers
        cfg.MODEL.MOTION_DECODER.NUM_DECODER_BLOCKS = args.num_layers
    if args.dropout is not None:
        cfg.MODEL.CONTEXT_ENCODER.DROPOUT_OF_ATTN = args.dropout
        cfg.MODEL.MOTION_DECODER.DROPOUT_OF_ATTN = args.dropout

    # 更新损失配置
    cfg.tied_noise = args.tied_noise
    cfg.LOSS_NN_MODE = args.loss_nn_mode
    cfg.LOSS_REG_REDUCTION = args.loss_reg_reduction

    # 更新数据配置
    cfg.MODEL.CONTEXT_ENCODER.AGENTS = args.max_num_agents
    cfg.rotate = args.rotate
    if args.rotate:
        cfg.rotate_time_frame = args.rotate_time_frame
    cfg.data_norm = args.data_norm

    # 更新优化配置
    if args.init_lr is not None:
        cfg.OPTIMIZATION.LR = args.init_lr
    if args.weight_decay is not None:
        cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay
    cfg.OPTIMIZATION.LOSS_WEIGHTS["cls"] = args.loss_cls_weight
    if args.epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
        cfg.val_batch_size = args.batch_size
        cfg.test_batch_size = args.batch_size * 2
    if args.checkpt_freq is not None:
        cfg.checkpt_freq = args.checkpt_freq
    cfg.max_num_ckpts = args.max_num_ckpts

    # 更新恢复配置
    cfg.RESUME.resume = args.resume
    cfg.RESUME.ckpt_name = args.ckpt_name
    cfg.RESUME.start_epoch = args.start_epoch
    cfg.RESUME.early_stop = args.early_stop

    # 为发布实验保持标签紧凑且稳定。
    k_value = cfg.K_LIST[-1] if len(cfg.K_LIST) > 0 else cfg.MODEL.NUM_PROPOSED_QUERY
    data_source_tag = {
        "original": "orig",
        "original_bal": "orig_bal",
    }[args.data_source]
    tag_parts = [
        tag_prefix or "run",
        args.fold_name,
        data_source_tag,
        cfg.denoising_method.upper(),
        cfg.MODEL.FUSER_NAME,
        f"K{k_value}",
        f"EP{cfg.OPTIMIZATION.NUM_EPOCHS}",
        f"BS{cfg.train_batch_size}",
        f"LR{cfg.OPTIMIZATION.LR}",
    ]
    tag = "_" + "_".join(str(x) for x in tag_parts)

    ### 创建保存目录 ###
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = cfg.create_dirs(tag_suffix=tag)

    """固定随机种子"""
    if args.fix_random_seed:
        set_random_seed(args.seed)

    """设置 tensorboard 和文本日志"""
    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, "../tb"))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

    """备份代码"""
    back_up_code_git(cfg, logger=logger)

    """打印配置文件"""
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def build_data_loader(cfg, args, mode="train"):
    """构建数据加载器
    
    Args:
        cfg: 配置对象
        args: 命令行参数
        mode: 模式，可选值为 "train" 或 "eval"
    
    Returns:
        数据加载器字典或测试数据加载器
    """

    DatasetClass = EgoTrajDataset
    collate_fn = seq_collate_egotraj

    def build_loader(cfg, args, split, batch_size, shuffle):
        """构建单个数据加载器
        
        Args:
            cfg: 配置对象
            args: 命令行参数
            split: 数据分割（train/val/test）
            batch_size: 批大小
            shuffle: 是否打乱数据
        
        Returns:
            数据加载器
        """
        dset = DatasetClass(
            cfg=cfg,
            split=split,
            data_dir=args.data_dir,
            rotate_time_frame=args.rotate_time_frame,
            type=args.data_source,
            source=args.fold_name,
        )
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return loader

    loader_dict = {}
    if mode == "eval":
        split_list = ["test"]
        batch_size_list = [cfg.test_batch_size]
        suffle_list = [False]
    elif mode == "train":
        split_list = ["train", "val", "test"]
        batch_size_list = [
            cfg.train_batch_size,
            cfg.val_batch_size,
            cfg.test_batch_size,
        ]
        suffle_list = [True, False, False]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    for split, batch_size, shuffle in zip(split_list, batch_size_list, suffle_list):
        loader = build_loader(cfg, args, split, batch_size, shuffle)
        loader_dict[split] = loader

    if mode == "eval":
        return loader_dict["test"]
    elif mode == "train":
        return loader_dict["train"], loader_dict["val"], loader_dict["test"]


def build_network(cfg, args, logger):
    """构建去噪模型网络
    
    Args:
        cfg: 配置对象
        args: 命令行参数
        logger: 日志记录器
    
    Returns:
        去噪器模型
    """
    # 构建 BiFlow 模型
    model = BiFlowModel(
        model_config=cfg.MODEL,
        logger=logger,
        config=cfg,
    )

    # 构建 Flow Matcher
    if cfg.denoising_method == "fm":
        denoiser = BiFlowMatcher(
            cfg,
            model,
            logger=logger,
        )
    else:
        raise NotImplementedError(
            f"Denoising method [{cfg.denoising_method}] is not implemented yet."
        )

    return denoiser


def prepare_train_context(args):
    """准备训练的运行环境、数据加载器和配置
    
    Args:
        args: 命令行参数
    
    Returns:
        cfg: 配置对象
        logger: 日志记录器
        tb_log: TensorBoard 日志记录器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 设置 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 如果未指定，根据 fold_name 自动设置 data_dir 和 max_num_agents
    if args.fold_name not in FOLD_CONFIG:
        raise ValueError(
            f"Invalid fold name: '{args.fold_name}'. "
            f"Expected one of: {list(FOLD_CONFIG.keys())}"
        )
    fold_cfg = FOLD_CONFIG[args.fold_name]
    if args.data_dir is None:
        args.data_dir = fold_cfg["data_dir"]
    if args.max_num_agents is None:
        args.max_num_agents = fold_cfg["max_num_agents"]

    tag_prefix = "v1"
    cfg, logger, tb_log = init_basics(args, tag_prefix=tag_prefix)
    train_loader, val_loader, test_loader = build_data_loader(cfg, args)
    cfg.save_updated_yml()  # 重新保存，包含来自数据集的归一化统计信息

    # 在模型初始化之前重新设置种子，以便权重初始化是确定性的
    # 无论数据加载消耗了多少随机操作
    if args.fix_random_seed:
        set_random_seed(args.seed)

    return cfg, logger, tb_log, train_loader, val_loader, test_loader


def main():
    """训练 BiFlow 模型的主函数"""

    # 解析命令行参数
    args = parse_config()
    # 准备训练上下文
    cfg, logger, tb_log, train_loader, val_loader, test_loader = prepare_train_context(args)

    # 构建网络
    denoiser = build_network(cfg, args, logger)

    # 创建训练器
    trainer = BiFlowTrainer(
        cfg=cfg,
        denoiser=denoiser,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
