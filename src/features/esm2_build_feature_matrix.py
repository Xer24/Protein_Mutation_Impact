from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import argparse
from src.features.esm2_embedding import ESM2Config, embed_sequences_esm2



def build_and_save(
    data_path: Path,
    out_dir: Path = Path("artifacts/features"),
    seq_col: str = "mutated_sequence",
    y_col: str = "DMS_score_bin",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)

    # Basic cleaning
    df = df.dropna(subset=[seq_col, y_col]).copy()

    seqs = df[seq_col].astype(str).tolist()

# dedupe but preserve order
    unique_seqs = list(dict.fromkeys(seqs))


    y = df[y_col].astype(int).to_numpy()

    cfg = ESM2Config(
        model_name="esm2_t6_8M_UR50D",
        cache_path=Path("artifacts/esm2_cache.joblib"),
        device="auto",
        batch_size=32,
        max_len=None,
    )

    E_unique = embed_sequences_esm2(unique_seqs, cfg=cfg)

# map back to full order
    idx = {s:i for i,s in enumerate(unique_seqs)}
    X = np.stack([E_unique[idx[s]] for s in seqs], axis=0)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    # Optional: save row ids so you can trace back
    if "DMS_id" in df.columns:
        np.save(out_dir / "dms_id.npy", df["DMS_id"].astype(str).to_numpy())

    print("Saved:")
    print(f"  X: {out_dir / 'X.npy'}  shape={X.shape} dtype={X.dtype}")
    print(f"  y: {out_dir / 'y.npy'}  shape={y.shape} dtype={y.dtype}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--data",
    type=str,
    default="data/processed/proteingym_gfp_sarkisyan2016.parquet",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/features",
        help="Output directory for X.npy and y.npy",
    )
    args = parser.parse_args()

    build_and_save(
        data_path=Path(args.data),
        out_dir=Path(args.out),
    )

