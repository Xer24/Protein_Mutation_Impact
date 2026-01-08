from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import argparse
from src.features.esm2_embedding import ESM2Config, embed_sequences_esm2


def get_seq_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Find WT and mutant sequence columns."""
    candidates = [
        ("target_seq", "mutated_sequence"),
        ("wt_sequence", "mut_sequence"),
        ("sequence", "mutated_sequence"),
    ]
    for wt_col, mut_col in candidates:
        if wt_col in df.columns and mut_col in df.columns:
            return wt_col, mut_col
    raise KeyError(f"Could not find WT/mutant columns. Available: {df.columns.tolist()}")


def build_and_save(
    data_path: Path,
    out_dir: Path = Path("data/processed"),
    y_col: str = "DMS_score_bin",
    use_delta: bool = True,  # NEW: toggle delta vs absolute
):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    
    # Get column names
    wt_col, mut_col = get_seq_cols(df)
    
    # Basic cleaning
    df = df.dropna(subset=[mut_col, y_col]).copy()

    # Get sequences
    if use_delta:
        # Need WT sequence for delta computation
        wt_seq = df[wt_col].iloc[0]  # Assuming single protein
        print(f"WT sequence length: {len(wt_seq)}")
    
    mut_seqs = df[mut_col].astype(str).tolist()
    y = df[y_col].astype(int).to_numpy()

    print(f"Total samples: {len(mut_seqs)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Dedupe sequences for efficient embedding
    unique_seqs = list(dict.fromkeys(mut_seqs))
    print(f"Unique sequences to embed: {len(unique_seqs)}")
    
    # ESM2 config
    cfg = ESM2Config(
    model_name="esm2_t12_35M_UR50D",  # ← Upgrade to 640 dims
    cache_path=Path("artifacts/esm2_cache.joblib"),
    device="auto",
    batch_size=8,  # Smaller batch for larger model
    max_len=None,
)

    # Embed unique sequences
    print("Embedding sequences...")
    if use_delta:
        # Embed WT + all unique mutants
        all_seqs_to_embed = [wt_seq] + unique_seqs
        all_embeddings = embed_sequences_esm2(all_seqs_to_embed, cfg=cfg)
        
        wt_emb = all_embeddings[0]  # First is WT
        unique_mut_embs = all_embeddings[1:]  # Rest are mutants
    else:
        # Just embed mutants
        unique_mut_embs = embed_sequences_esm2(unique_seqs, cfg=cfg)

    # Map back to full order
    print("Mapping embeddings...")
    idx = {s: i for i, s in enumerate(unique_seqs)}
    mut_embs = np.stack([unique_mut_embs[idx[s]] for s in mut_seqs], axis=0)
    
    # Compute features
    if use_delta:
        print("Computing delta embeddings...")
        X = mut_embs - wt_emb[None, :]  # Delta: mutant - WT
        feature_type = "delta"
    else:
        print("Using absolute embeddings...")
        X = mut_embs  # Just mutant embeddings
        feature_type = "absolute"

    # Save
    np.save(out_dir / "gfp_delta_X.npy", X)  # ← Added "gfp_delta_"
    np.save(out_dir / "gfp_y.npy", y)        # ← Added "gfp_"

    # Save metadata
    meta = {
        "feature_type": feature_type,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "class_distribution": np.bincount(y).tolist(),
        "model_name": cfg.model_name,
    }
    
    import json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: save identifiers
    if "DMS_id" in df.columns:
        np.save(out_dir / "dms_id.npy", df["DMS_id"].astype(str).to_numpy())
    if "mutant" in df.columns:
        np.save(out_dir / "mutant.npy", df["mutant"].astype(str).to_numpy())

    print("\n" + "="*60)
    print("✅ SAVED:")
    print(f"  X: {out_dir / 'gfp_delta_X.npy'}")
    print(f"     Shape: {X.shape}")
    print(f"     Type: {feature_type} embeddings")
    print(f"  y: {out_dir / 'gfp_y.npy'}")
    print(f"     Shape: {y.shape}")
    print(f"     Class 0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"     Class 1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/proteingym_gfp_sarkisyan2016.parquet",
        help="Input parquet file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/features",
        help="Output directory for gfp_delta_X.npy and gfp_y.npy"
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute embeddings instead of deltas (not recommended)"
    )
    args = parser.parse_args()

    build_and_save(
        data_path=Path(args.data),
        out_dir=Path(args.out),
        use_delta=not args.absolute,  # Default is delta
    )