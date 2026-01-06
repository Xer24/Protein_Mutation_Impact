from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_mutation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse mutation like 'A16V' into from_aa='A', pos=16, to_aa='V'."""
    s = df["mutation"].astype(str)
    df = df.copy()
    df["from_aa"] = s.str[0]
    df["to_aa"] = s.str[-1]
    df["pos"] = s.str[1:-1].astype(int)
    return df


def main():
    pred_path = Path("artifacts/predictions/predictions_k836.parquet")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path.resolve()}")

    df = pd.read_parquet(pred_path)
    df = add_mutation_columns(df)

    print("Loaded:", pred_path)
    print("Rows:", len(df))
    print(df.head())

    # 1) Histogram of predicted probabilities
    plt.figure()
    plt.hist(df["prob_deleterious"].astype(float), bins=30)
    plt.xlabel("Predicted deleterious probability")
    plt.ylabel("Count")
    plt.title("Distribution of predictions (k sample)")
    plt.tight_layout()
    plt.show()

    # 2) Scatter: delta norm vs predicted probability
    plt.figure()
    plt.scatter(df["delta_l2"].astype(float), df["prob_deleterious"].astype(float), alpha=0.7)
    plt.xlabel("||Δ embedding|| (L2)")
    plt.ylabel("Predicted deleterious probability")
    plt.title("Representation shift vs predicted harm")
    plt.tight_layout()
    plt.show()

    # 3) Top-N bar chart
    top_n = 25
    df_top = df.sort_values("prob_deleterious", ascending=False).head(top_n)

    plt.figure()
    plt.bar(df_top["mutation"], df_top["prob_deleterious"].astype(float))
    plt.xticks(rotation=90)
    plt.ylabel("Predicted deleterious probability")
    plt.title(f"Top {top_n} most deleterious predicted (sample)")
    plt.tight_layout()
    plt.show()

    # 4) Position-wise summary (mean prob per position in your sample)
    pos_stats = (
        df.groupby("pos", as_index=False)
          .agg(mean_prob=("prob_deleterious", "mean"),
               max_prob=("prob_deleterious", "max"),
               mean_delta=("delta_l2", "mean"),
               n=("mutation", "count"))
          .sort_values("pos")
    )

    plt.figure()
    plt.plot(pos_stats["pos"], pos_stats["mean_prob"])
    plt.xlabel("Position (1-indexed)")
    plt.ylabel("Mean predicted deleterious probability")
    plt.title("Position sensitivity (mean over sampled mutations)")
    plt.tight_layout()
    plt.show()

    # Optional: print most sensitive positions (by mean_prob, but require enough samples)
    min_n = 3
    hot = pos_stats[pos_stats["n"] >= min_n].sort_values("mean_prob", ascending=False).head(10)
    print("\nTop positions by mean predicted deleterious prob (requires n>=3 at position):")
    print(hot)


if __name__ == "__main__":
    main()
