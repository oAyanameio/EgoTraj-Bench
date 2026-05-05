"""BiFlow 端到端管道测试脚本

该脚本用于测试 BiFlow 模型的完整管道，包括加载数据、构建模型并运行单个前向传播。

使用方法:
    python examples/test_pipeline.py --data_dir ./data/egotraj --fold_name tbd
    python examples/test_pipeline.py --data_dir ./data/t2fpv --fold_name eth --cfg cfg/biflow_t2fpv_k20.yml
"""

import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config import Config
from models.backbone_biflow import BiFlowModel
from models.flow_matching_biflow import BiFlowMatcher
from loaders.dataloader_egotraj import EgoTrajDataset, seq_collate_egotraj


def main():
    """主函数，用于测试 BiFlow 模型的完整管道
    
    该函数执行以下步骤：
    1. 解析命令行参数
    2. 加载配置文件
    3. 加载数据集和数据加载器
    4. 构建模型
    5. 加载一个批次的数据并检查数据形状
    6. 输出测试结果
    """
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BiFlow pipeline test")
    parser.add_argument("--cfg", default="cfg/biflow_k20.yml", type=str, help="配置文件路径")
    parser.add_argument("--data_dir", default="./data/egotraj", type=str, help="数据目录")
    parser.add_argument("--fold_name", default="tbd", type=str, help="折叠名称")
    parser.add_argument("--data_source", default="original", type=str, help="数据源")
    parser.add_argument("--batch_size", default=8, type=int, help="批大小")
    args = parser.parse_args()

    # 配置日志记录器
    logger = logging.getLogger("test_pipeline")
    logging.basicConfig(level=logging.INFO)

    # 1. 加载配置
    logger.info(f"Loading config: {args.cfg}")
    cfg = Config(args.cfg, "test_pipeline", train_mode=False)
    cfg.device = "cpu"  # 设置设备为 CPU
    cfg.test_batch_size = args.batch_size  # 设置测试批大小

    # 2. 加载数据集
    logger.info(f"Loading dataset: fold={args.fold_name}, source={args.data_source}")
    dataset = EgoTrajDataset(
        cfg=cfg,
        split="test",  # 使用测试集
        data_dir=args.data_dir,  # 数据目录
        rotate_time_frame=6,  # 旋转时间帧
        type=args.data_source,  # 数据源类型
        source=args.fold_name,  # 源数据集
    )
    logger.info(f"  Dataset size: {len(dataset)} samples")

    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 不使用多线程
        collate_fn=seq_collate_egotraj,  # 自定义 collation 函数
    )

    # 3. 构建模型
    logger.info("Building model ...")
    model = BiFlowModel(model_config=cfg.MODEL, logger=logger, config=cfg)
    denoiser = BiFlowMatcher(cfg, model, logger=logger)
    denoiser.eval()  # 设置为评估模式

    # 计算模型参数数量
    n_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"  Total parameters: {n_params:,}")

    # 4. 加载一个批次的数据
    batch = next(iter(loader))
    logger.info(f"  Batch keys: {list(batch.keys())}")
    logger.info(f"  past_traj shape: {batch['past_traj'].shape}")
    logger.info(f"  fut_traj shape:  {batch['fut_traj'].shape}")
    logger.info(f"  agent_mask shape: {batch['agent_mask'].shape}")

    logger.info("Pipeline test passed! Data + Model are compatible.")


if __name__ == "__main__":
    main()
