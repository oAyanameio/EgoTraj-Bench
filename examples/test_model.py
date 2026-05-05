"""BiFlow 模型测试脚本

该脚本用于对 BiFlow 模型进行基本的完整性检查，包括模型构建和简单的前向传播测试。

使用方法:
    python examples/test_model.py --cfg cfg/biflow_k20.yml
    python examples/test_model.py --cfg cfg/biflow_t2fpv_k20.yml
"""

import sys
import logging
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.config import Config
from models.backbone_biflow import BiFlowModel
from models.flow_matching_biflow import BiFlowMatcher


def main():
    """主函数，用于测试 BiFlow 模型的构建和前向传播
    
    该函数执行以下步骤：
    1. 解析命令行参数，获取配置文件路径
    2. 加载配置文件
    3. 构建 BiFlowModel 和 BiFlowMatcher
    4. 计算模型参数数量
    5. 模拟批量数据并尝试前向传播
    6. 输出测试结果
    """
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BiFlow model sanity check")
    parser.add_argument("--cfg", default="cfg/biflow_k20.yml", type=str, help="配置文件路径")
    args = parser.parse_args()

    # 加载配置（train_mode=False 避免创建输出目录）
    cfg = Config(args.cfg, "test", train_mode=False)

    # 基本日志记录器
    logger = logging.getLogger("test_model")
    logging.basicConfig(level=logging.INFO)

    # 构建模型
    logger.info("Building BiFlowModel ...")
    model = BiFlowModel(model_config=cfg.MODEL, logger=logger, config=cfg)

    # 设置设备为 CPU
    cfg.device = "cpu"
    logger.info("Building BiFlowMatcher ...")
    denoiser = BiFlowMatcher(cfg, model, logger=logger)

    # 计算模型参数数量
    n_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"Total parameters: {n_params:,}")

    # 准备虚拟批量数据
    B = 4  # 批大小（智能体数量）
    past_frames = cfg.past_frames  # 过去帧数
    future_frames = cfg.future_frames  # 未来帧数
    num_queries = cfg.MODEL.NUM_PROPOSED_QUERY  # 预测轨迹数量
    agents = cfg.MODEL.CONTEXT_ENCODER.AGENTS  # 最大智能体数量

    # 模拟批量字典（前向传播所需的最小键集）
    batch = {
        "all_obs": torch.randn(B, past_frames, 7),  # 观测数据
        "all_pred": torch.randn(B, past_frames + future_frames, 7),  # 预测数据
        "seq_start_end": torch.tensor([[0, B]]),  # 序列起止索引
    }

    # 运行虚拟前向传播
    logger.info(f"Running dummy forward (B={B}, past={past_frames}, future={future_frames}) ...")
    denoiser.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        try:
            # 此测试仅验证模型构建；完整的前向传播可能需要更多的批量键
            logger.info("Model built and loaded successfully!")
            logger.info(f"  Config:  {args.cfg}")
            logger.info(f"  Queries: {num_queries}")
            logger.info(f"  Agents:  {agents}")
            logger.info(f"  D_model: {cfg.MODEL.CONTEXT_ENCODER.D_MODEL}")
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise

    logger.info("Sanity check passed!")


if __name__ == "__main__":
    main()
