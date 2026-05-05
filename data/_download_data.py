"""从 HuggingFace 下载 EgoTraj-Bench 数据集

该脚本从 HuggingFace 仓库下载 EgoTraj-Bench 数据集，并按照预期的目录结构组织数据。

主要功能：
- 从 HuggingFace 仓库下载 EgoTraj-TBD 和 T2FPV-ETH 数据集
- 自动安装必要的依赖（huggingface_hub）
- 创建所需的目录结构
- 下载并保存数据集文件
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

# HuggingFace 仓库 ID
HF_REPO = "ZoeyLIU1999/EgoTraj-Bench"
# 默认数据目录
DEFAULT_DATA_DIR = Path("./data")


def main():
    """主函数，执行数据集下载流程
    
    解析命令行参数，创建目录结构，从 HuggingFace 下载数据集文件。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Download EgoTraj-Bench from HuggingFace")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory for organized output data (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    print("=== Downloading EgoTraj-Bench from HuggingFace (direct mode) ===")

    # 尝试导入 huggingface_hub，如果未安装则自动安装
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

    from huggingface_hub import HfApi, hf_hub_url

    # 创建数据目录
    egotraj_dir = data_dir / "egotraj"  # EgoTraj-TBD 数据集目录
    t2fpv_dir = data_dir / "t2fpv"  # T2FPV-ETH 数据集目录
    egotraj_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
    t2fpv_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）

    # 初始化 HuggingFace API
    api = HfApi()
    # 列出仓库中的所有文件
    repo_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")

    # 数据集映射：(远程路径前缀, 本地目标目录, 数据集名称)
    mappings = [
        ("L2-processed/EgoTraj-TBD/", egotraj_dir, "EgoTraj-TBD"),
        ("L2-processed/T2FPV-ETH/", t2fpv_dir, "T2FPV-ETH"),
    ]

    # 下载每个数据集
    for prefix, target_dir, dataset_name in mappings:
        # 筛选匹配的文件（以指定前缀开头且以 .npz 结尾）
        matched_files = [
            remote_path
            for remote_path in repo_files
            if remote_path.startswith(prefix) and remote_path.endswith(".npz")
        ]
        if not matched_files:
            print(f"  Warning: no files found for {dataset_name} under {prefix}")
            continue

        print(f"\nDownloading {dataset_name} -> {target_dir}/")
        # 下载每个匹配的文件
        for remote_path in matched_files:
            # 构建本地文件路径
            local_path = target_dir / Path(remote_path).name
            # 生成 HuggingFace 文件 URL
            file_url = hf_hub_url(
                repo_id=HF_REPO,
                filename=remote_path,
                repo_type="dataset",
            )
            # 下载文件
            urllib.request.urlretrieve(file_url, local_path)
            print(f"  saved: {local_path.name}")

    # 打印完成信息
    print("\n=== Done! ===")
    print("Data directory structure:")
    print(f"  {data_dir}/")
    print(f"  ├── egotraj/    (EgoTraj-TBD .npz files)")
    print(f"  └── t2fpv/      (T2FPV-ETH .npz files)")


if __name__ == "__main__":
    """脚本入口点"""
    main()
