"""Split listings into train/val/test by label_group.

Splitting by row would put the same product in train and test and inflate every
metric, so whole label groups go to exactly one split.

    python -m recclens.data.split
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

SPLITS = ("train", "val", "test")


def split_by_group(df: pd.DataFrame, ratios: list[float], seed: int) -> dict[str, pd.DataFrame]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    groups = df["label_group"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_train = int(round(ratios[0] * len(groups)))
    n_val = int(round(ratios[1] * len(groups)))
    assignment = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }
    return {
        name: df[df["label_group"].isin(set(ids))].reset_index(drop=True)
        for name, ids in assignment.items()
    }


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    df = pd.read_csv(root / cfg.data.train_csv)
    parts = split_by_group(df, list(cfg.data.split.ratios), cfg.data.split.seed)

    out_dir = root / cfg.data.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = parts[name]
        part.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, {part['label_group'].nunique()} groups")


if __name__ == "__main__":
    main()
