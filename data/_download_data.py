"""
Download EgoTraj-Bench dataset from HuggingFace
into the expected directory structure.
"""

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

HF_REPO = "ZoeyLIU1999/EgoTraj-Bench"
DEFAULT_DATA_DIR = Path("./data")


def main():
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

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])

    from huggingface_hub import HfApi, hf_hub_url

    egotraj_dir = data_dir / "egotraj"
    t2fpv_dir = data_dir / "t2fpv"
    egotraj_dir.mkdir(parents=True, exist_ok=True)
    t2fpv_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")

    mappings = [
        ("L2-processed/EgoTraj-TBD/", egotraj_dir, "EgoTraj-TBD"),
        ("L2-processed/T2FPV-ETH/", t2fpv_dir, "T2FPV-ETH"),
    ]

    for prefix, target_dir, dataset_name in mappings:
        matched_files = [
            remote_path
            for remote_path in repo_files
            if remote_path.startswith(prefix) and remote_path.endswith(".npz")
        ]
        if not matched_files:
            print(f"  Warning: no files found for {dataset_name} under {prefix}")
            continue

        print(f"\nDownloading {dataset_name} -> {target_dir}/")
        for remote_path in matched_files:
            local_path = target_dir / Path(remote_path).name
            file_url = hf_hub_url(
                repo_id=HF_REPO,
                filename=remote_path,
                repo_type="dataset",
            )
            urllib.request.urlretrieve(file_url, local_path)
            print(f"  saved: {local_path.name}")

    print("\n=== Done! ===")
    print("Data directory structure:")
    print(f"  {data_dir}/")
    print(f"  ├── egotraj/    (EgoTraj-TBD .npz files)")
    print(f"  └── t2fpv/      (T2FPV-ETH .npz files)")


if __name__ == "__main__":
    main()
