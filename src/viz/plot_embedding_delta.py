"""
plot_embedding_deltas.py

Plot per-position embedding deltas produced by delta_features.py.

For each mutation:
- Line plot of per-position embedding delta
- Vertical line marking mutation position
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_delta_file(in_dir: Path, mutation: str) -> Dict[str, Any]:
    path = in_dir / f"per_position_delta_{mutation.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def plot_single_mutation(
    mutation: str,
    per_pos: List[float],
    mut_pos_1idx: int,
    out_path: Path,
) -> None:
    L = len(per_pos)
    x = range(1, L + 1)

    plt.figure()
    plt.plot(x, per_pos)
    plt.axvline(mut_pos_1idx, linestyle="--")
    plt.xlabel("Residue position (1-indexed)")
    plt.ylabel("Per-position embedding delta (L2)")
    plt.title(f"Per-position embedding delta: {mutation} (pos {mut_pos_1idx})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overlay(series: List[Dict[str, Any]], out_path: Path, max_mutations: int = 8) -> None:
    plt.figure()

    for item in series[:max_mutations]:
        per_pos = item["per_position_l2"]
        L = len(per_pos)
        x = range(1, L + 1)
        plt.plot(x, per_pos, label=item["mutation"])

    plt.xlabel("Residue position (1-indexed)")
    plt.ylabel("Per-position embedding delta (L2)")
    plt.title("Per-position embedding deltas (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="artifacts/embedding_deltas")
    p.add_argument("--out_dir", type=str, default="artifacts/figures")
    p.add_argument("--mutations", type=str, nargs="*", default=None)
    p.add_argument("--overlay", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect mutations if not provided
    if not args.mutations:
        files = sorted(in_dir.glob("per_position_delta_*.json"))
        if not files:
            raise FileNotFoundError(
                f"No per_position_delta_*.json files found in {in_dir}"
            )
        mutations = [f.stem.replace("per_position_delta_", "") for f in files]
    else:
        mutations = [m.upper() for m in args.mutations]

    loaded = []

    for mut in mutations:
        obj = load_delta_file(in_dir, mut)
        mutation = obj["mutation"].upper()
        mut_pos = int(obj["mutant_position_1idx"])
        per_pos = obj["per_position_l2"]

        loaded.append(
            {
                "mutation": mutation,
                "mutant_position_1idx": mut_pos,
                "per_position_l2": per_pos,
            }
        )

        out_path = out_dir / f"delta_{mutation}.png"
        plot_single_mutation(
            mutation=mutation,
            per_pos=per_pos,
            mut_pos_1idx=mut_pos,
            out_path=out_path,
        )
        print(f"Saved: {out_path}")

    if args.overlay and len(loaded) >= 2:
        overlay_path = out_dir / "delta_overlay.png"
        plot_overlay(loaded, overlay_path)
        print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
