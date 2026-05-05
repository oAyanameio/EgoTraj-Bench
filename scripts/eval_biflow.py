"""BiFlow 模型评估脚本

该脚本用于评估 BiFlow 轨迹预测模型，包括参数解析、配置初始化、数据加载器构建、
网络构建和评估过程等功能。
"""

import os
import torch
import argparse
import copy
from glob import glob
from pathlib import Path
import sys

from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
print(f"Using root directory: {ROOT_DIR}")

from loaders.dataloader_egotraj import EgoTrajDataset, seq_collate_egotraj

from utils.config import Config
from utils.utils import set_random_seed, log_config_to_file

from models.flow_matching_biflow import BiFlowMatcher
from models.backbone_biflow import BiFlowModel
from trainer.biflow_trainer import BiFlowTrainer


def parse_config():
    """解析命令行参数并返回配置选项
    
    Returns:
        解析后的命令行参数
    """

    parser = argparse.ArgumentParser()

    # 基本配置
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="加载模型的检查点目录。",
    )
    parser.add_argument("--cfg", default="auto", type=str, help="配置文件路径")
    parser.add_argument(
        "--exp",
        default="tbd_eval",
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
        "--batch_size",
        default=512,
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

    # 可复现性配置
    parser.add_argument(
        "--fix_random_seed",
        action="store_true",
        default=False,
        help="固定随机种子以确保可复现性",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9527,
        help="设置随机种子以分割测试集用于训练评估。",
    )

    ### FM 参数 ###
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=100,
        help="FlowMatcher 的采样时间步数。",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lin_poly",
        choices=["euler", "lin_poly"],
        help="FlowMatcher 的求解器。",
    )
    parser.add_argument(
        "--lin_poly_p",
        type=int,
        default=5,
        help="线性阶段多项式的次数。",
    )
    parser.add_argument(
        "--lin_poly_long_step",
        type=int,
        default=1000,
        help="在线性阶段模拟斜率的步数。",
    )
    ### FM 参数 ###

    ### 评估模型 ###
    parser.add_argument(
        "--mode",
        type=str,
        default="75",
        help="评估模式: epoch 编号（例如，'75'）、'best' 或 'last'。",
    )
    parser.add_argument(
        "--save_for_vis",
        default=False,
        action="store_true",
        help="保存预测结果用于可视化。",
    )

    return parser.parse_args()


def init_basics(args):
    """初始化实验的基本配置
    
    Args:
        args: 命令行参数
    
    Returns:
        cfg: 配置对象
        logger: 日志记录器
        tb_log: TensorBoard 日志记录器
    """

    """加载配置文件"""
    result_dir = args.ckpt_dir
    if args.cfg == "auto":
        yml_ls = glob(result_dir + "/*.yml")
        print(f"result_dir: {result_dir}")
        assert (
            len(yml_ls) >= 1
        ), "At least one config file should be found in the directory."
        yml_path = [f for f in yml_ls if "_updated.yml" in os.path.basename(f)][0]
        args.cfg = yml_path
    cfg = Config(args.cfg, f"{args.exp}", train_mode=False)

    tag = "_"
    if args.fold_name != "tbd":
        tag += f"{args.fold_name}_"

    ### 更新数据版本 ###
    if args.data_source == "original":
        tag += "orig_"
    elif args.data_source == "gt_matching":
        tag += "gt_mat_"
    elif args.data_source == "occ_rep":
        tag += "occ_rep_"
    elif args.data_source == "original_bal":
        tag += "orig_bal_"
    else:
        raise ValueError(f"Invalid data source: {args.data_source}")

    ### 更新 FM 参数 ###
    def _update_fm_params(args, cfg, tag):
        if cfg.denoising_method == "fm":
            cfg.sampling_steps = args.sampling_steps
            cfg.solver = args.solver

            if args.solver == "euler":
                solver_tag_ = args.solver
            elif args.solver == "lin_poly":
                cfg.lin_poly_p = args.lin_poly_p
                cfg.lin_poly_long_step = args.lin_poly_long_step
                solver_tag_ = (
                    f"lin_poly_p{args.lin_poly_p}_long{args.lin_poly_long_step}"
                )

            fm_tag_ = f"FM_S{cfg.sampling_steps}_{solver_tag_}"
            tag += fm_tag_
            cfg.solver_tag = fm_tag_

        return cfg, tag

    cfg, tag = _update_fm_params(args, cfg, tag)

    def _update_optimization_params(args, cfg, tag):
        if args.batch_size is not None:
            # 覆盖批大小
            cfg.train_batch_size = args.batch_size
            cfg.val_batch_size = args.batch_size
            cfg.test_batch_size = args.batch_size
        return cfg, tag

    cfg, tag = _update_optimization_params(args, cfg, tag)

    ### 创建保存目录 ###
    tag += "_test_set"

    tag += f"_{args.mode}"
    tag = tag.replace("__", "_")
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = cfg.create_dirs(tag_suffix=tag)

    """固定随机种子"""
    if args.fix_random_seed:
        set_random_seed(args.seed)

    """设置 tensorboard 和文本日志"""
    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, "../tb_eval"))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)

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
        dset = EgoTrajDataset(
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
            collate_fn=seq_collate_egotraj,
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


def prepare_eval_context(args):
    """准备评估的运行环境、配置、加载器和评估模式
    
    Args:
        args: 命令行参数
    
    Returns:
        cfg: 配置对象
        logger: 日志记录器
        tb_log: TensorBoard 日志记录器
        test_loader: 测试数据加载器
        eval_mode: 评估模式
    """

    # 设置 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 如果未指定，根据 fold_name 自动设置 data_dir
    if args.data_dir is None:
        if args.fold_name == "tbd":
            args.data_dir = "./data/egotraj"
        else:
            args.data_dir = "./data/t2fpv"

    assert args.ckpt_dir is not None, "Must specify --ckpt_dir for evaluation."

    # 解析模式: "best", "last", 或整数 epoch 编号
    eval_mode = args.mode
    if eval_mode not in ("best", "last"):
        eval_mode = int(eval_mode)

    # 初始化基本配置
    cfg, logger, tb_log = init_basics(args)

    # 检查配置中是否有归一化参数
    if cfg.get("fut_traj_min", None) is None:
        # 旧检查点没有 yml 中的归一化参数 — 必须从训练数据计算
        _train_loader, _val_loader, test_loader = build_data_loader(
            cfg, args, mode="train"
        )
    else:
        test_loader = build_data_loader(cfg, args, mode="eval")

    return cfg, logger, tb_log, test_loader, eval_mode


def main():
    """评估 BiFlow 模型的主函数"""

    # 解析命令行参数
    args = parse_config()
    # 准备评估上下文
    cfg, logger, tb_log, test_loader, eval_mode = prepare_eval_context(args)

    # 构建网络
    denoiser = build_network(cfg, args, logger)

    # 创建训练器
    trainer = BiFlowTrainer(
        cfg=cfg,
        denoiser=denoiser,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        tb_log=tb_log,
        logger=logger,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        ema_update_every=1,
    )

    # 执行测试
    trainer.test(mode=eval_mode, save_for_vis=args.save_for_vis)
    # 清空 GPU 缓存
    torch.cuda.empty_cache()
    # 打印评估信息
    print("--------------------------------")
    print("ckpt_dir: ", args.ckpt_dir)
    print("data_source: ", args.data_source)
    print("fold_name: ", args.fold_name)
    print("solver: ", args.solver)
    print("--------------------------------")


if __name__ == "__main__":
    main()
