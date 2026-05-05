"""数据集和折叠配置注册表

该模块定义了不同数据集的配置信息，包括数据目录和最大智能体数量。
当在命令行参数中未明确指定时，train_biflow.py 会使用这些配置自动设置 data_dir 和 max_num_agents。

主要配置项：
- FOLD_CONFIG: 字典，映射折叠名称到数据集配置
  - key: 折叠名称（如 "tbd", "eth", "hotel" 等）
  - value: 包含以下键的字典
    - data_dir: 数据集目录路径
    - max_num_agents: 数据集中的最大智能体数量
"""

# 数据集和折叠配置注册表
# 映射折叠名称 -> {数据目录, 最大智能体数量}
# 当在命令行参数中未明确指定时，train_biflow.py 会使用这些配置自动设置 data_dir 和 max_num_agents

FOLD_CONFIG = {
    # EgoTraj-TBD 数据集
    "tbd": {
        "data_dir": "./data/egotraj",  # 数据目录路径
        "max_num_agents": 16,  # 最大智能体数量
    },
    # T2FPV-ETH 数据集（五个留一法折叠）
    "eth": {
        "data_dir": "./data/t2fpv",  # 数据目录路径
        "max_num_agents": 32,  # 最大智能体数量
    },
    "hotel": {
        "data_dir": "./data/t2fpv",  # 数据目录路径
        "max_num_agents": 32,  # 最大智能体数量
    },
    "univ": {
        "data_dir": "./data/t2fpv",  # 数据目录路径
        "max_num_agents": 32,  # 最大智能体数量
    },
    "zara1": {
        "data_dir": "./data/t2fpv",  # 数据目录路径
        "max_num_agents": 32,  # 最大智能体数量
    },
    "zara2": {
        "data_dir": "./data/t2fpv",  # 数据目录路径
        "max_num_agents": 32,  # 最大智能体数量
    },
}
