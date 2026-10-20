"""Zero-shot baselines: SigLIP 2 on images, and SigLIP 2 + e5 concatenated.

Runs recclens.embed then recclens.evaluate for each, and prints a results table.

    python -m recclens.baselines.zero_shot --queries data/processed/queries.csv
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASELINES = {
    "zero_shot_siglip": ["model.mode=image"],
    "zero_shot_siglip_e5": ["model.mode=concat"],
}


def run(module: str, overrides: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *overrides], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baselines"))
    args = parser.parse_args()

    results = {}
    for name, overrides in BASELINES.items():
        run_dir = args.out_dir / name
        shared = [f"eval.split={args.split}"]
        if args.queries:
            shared.append(f"eval.queries_csv={args.queries}")
        run("recclens.embed", [*overrides, *shared, f"hydra.run.dir={run_dir}"])

        eval_args = [*shared, f"eval.item_embeddings={run_dir / 'items.npy'}"]
        if args.queries and (run_dir / "queries.npy").exists():
            eval_args.append(f"eval.query_embeddings={run_dir / 'queries.npy'}")
        run("recclens.evaluate", [*eval_args, f"hydra.run.dir={run_dir}"])
        results[name] = json.loads((run_dir / "metrics.json").read_text())

    keys = ["match_f1", "item_recall@10", "query_recall@10", "query_ndcg@10"]
    print("\n| baseline | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for name, metrics in results.items():
        cells = [f"{metrics[k]:.4f}" if k in metrics else "-" for k in keys]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
