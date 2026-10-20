"""Download the Shopee Price Match data from Kaggle.

The competition data is not redistributed in this repo. You need a Kaggle
account, the competition rules accepted, and ~/.kaggle/kaggle.json in place.

    python -m recclens.data.download
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def download(competition: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "train.csv").exists():
        print(f"{out_dir / 'train.csv'} already exists, skipping download")
        return
    cli = [sys.executable, "-m", "kaggle", "competitions", "download"]
    subprocess.run([*cli, "-c", competition, "-p", str(out_dir)], check=True)
    archive = out_dir / f"{competition}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    archive.unlink()
    print(f"Extracted to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="shopee-product-matching")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.competition, args.out_dir)


if __name__ == "__main__":
    main()
