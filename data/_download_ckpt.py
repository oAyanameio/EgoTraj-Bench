"""从 HuggingFace 数据集仓库下载发布的模型 checkpoint

该脚本从 HuggingFace 仓库下载 EgoTraj-Bench 项目的预训练模型 checkpoint，并按照预期的目录结构组织文件。

预期的远程布局：
  models/
    README.md
    T2FPV-eth/
      config_updated.yml
      models/checkpoint_best.pt
    T2FPV-hotel/
      ...
    T2FPV-univ/
      ...
    T2FPV-zara1/
      ...
    T2FPV-zara2/
      ...
    EgoTraj-TBD/
      ...

本地输出布局（默认）：
  checkpoints/<ModelName>/
    config_updated.yml
    models/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import urllib.request

from huggingface_hub import HfApi, hf_hub_url


# 默认 HuggingFace 仓库 ID
DEFAULT_REPO_ID = "ZoeyLIU1999/EgoTraj-Bench"
# 默认输出根目录
DEFAULT_OUTPUT_ROOT = Path("./checkpoints")

# 默认要下载的模型版本列表
DEFAULT_RELEASES = [
    "T2FPV-eth",
    "T2FPV-hotel",
    "T2FPV-univ",
    "T2FPV-zara1",
    "T2FPV-zara2",
    "EgoTraj-TBD",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 包含命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(
        description="Download EgoTraj-Bench release checkpoints from HuggingFace"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Local checkpoint root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for gated/private repos (optional for public repo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files",
    )
    return parser.parse_args()


def release_files(release_name: str) -> Iterable[tuple[str, Path]]:
    """生成指定版本的文件路径对
    
    生成远程仓库中的文件路径和本地相对路径的配对。
    
    Args:
        release_name: 版本名称
    
    Yields:
        tuple[str, Path]: (远程仓库中的文件路径, 本地相对路径)
    """
    # 配置文件
    yield (
        f"models/{release_name}/config_updated.yml",
        Path("config_updated.yml"),
    )
    # 模型 checkpoint 文件
    yield (
        f"models/{release_name}/models/checkpoint_best.pt",
        Path("models/checkpoint_best.pt"),
    )


def download_file_to_path(url: str, dst_path: Path, token: str | None) -> None:
    """下载文件到指定路径
    
    Args:
        url: 文件下载 URL
        dst_path: 目标文件路径
        token: HuggingFace 令牌（用于访问私有仓库）
    """
    headers = {}
    # 如果提供了令牌，则添加到请求头
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 创建请求对象
    request = urllib.request.Request(url=url, headers=headers)
    # 下载文件并写入目标路径
    with urllib.request.urlopen(request) as response, dst_path.open("wb") as out_f:
        while True:
            # 分块读取和写入
            chunk = response.read(1024 * 1024)  # 1MB 块
            if not chunk:
                break
            out_f.write(chunk)


def main() -> None:
    """主函数，执行模型 checkpoint 下载流程
    
    解析命令行参数，列出仓库文件，下载指定版本的模型 checkpoint。
    """
    # 解析命令行参数
    args = parse_args()
    output_root: Path = args.output_root

    # 获取要下载的版本列表
    releases = list(DEFAULT_RELEASES)

    # 打印基本信息
    print("Source repo:", args.repo_id)
    print("Output root:", output_root.resolve())
    print("Planned releases:", ", ".join(releases))

    # 初始化 HuggingFace API
    api = HfApi(token=args.token)
    # 获取仓库中的所有文件
    repo_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))

    # 干运行模式：只打印计划下载的文件，不实际下载
    if args.dry_run:
        print("\nDry-run planned files:")
        for release_name in releases:
            for remote_path, rel_local in release_files(release_name):
                status = "OK" if remote_path in repo_files else "MISSING"
                print(
                    f"  [{status}] {remote_path} -> "
                    f"{(output_root / release_name / rel_local).as_posix()}"
                )
        return

    # 创建输出根目录
    output_root.mkdir(parents=True, exist_ok=True)

    # 下载每个版本的文件
    for release_name in releases:
        print(f"\n=== Downloading {release_name} ===")
        release_dir = output_root / release_name
        for remote_path, rel_local in release_files(release_name):
            # 检查文件是否存在于仓库中
            if remote_path not in repo_files:
                print(f"  [SKIP] Missing on HF: {remote_path}")
                continue

            # 构建本地目标路径
            local_dst = release_dir / rel_local
            # 创建父目录
            local_dst.parent.mkdir(parents=True, exist_ok=True)

            # 生成文件下载 URL
            file_url = hf_hub_url(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=remote_path,
            )
            # 下载文件
            download_file_to_path(file_url, local_dst, args.token)
            print(f"  saved: {local_dst}")

    # 打印完成信息
    print("\nDone.")


if __name__ == "__main__":
    """脚本入口点"""
    main()
