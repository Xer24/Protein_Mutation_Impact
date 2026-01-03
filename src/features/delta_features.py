"""
delta_features.py

Compute:
1) Global embedding distance between WT and mutant (cosine + L2 on pooled embedding)
2) Per-position embedding deltas (L2 norm per residue position)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch

try:
    import esm
except ImportError as e:
    raise ImportError(
        "Missing 'esm' package. Install with:\n"
        "  pip install fair-esm\n"
        "and ensure torch is installed.\n"
    ) from e


# ----------------------------
# Paths / Defaults
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "raw" / "proteingym_dms_substitutions.parquet"


@dataclass
class DeltaResult:
    mutation: str
    mutant_position_1idx: int
    global_cosine_distance: float
    global_l2_distance: float
    per_position_l2: List[float]  # length = sequence length


# ----------------------------
# Mutation parsing / WT reconstruction
# ----------------------------
def parse_mutation(mut: str) -> Tuple[str, int, str]:
    """
    Parse A23V -> (wt='A', pos_1idx=23, mut='V')
    """
    mut = mut.strip().upper()
    if len(mut) < 3:
        raise ValueError(f"Bad mutation string: {mut}")

    wt_aa = mut[0]
    mut_aa = mut[-1]
    pos_str = mut[1:-1]
    if not pos_str.isdigit():
        raise ValueError(f"Bad mutation position in: {mut}")

    pos_1idx = int(pos_str)
    return wt_aa, pos_1idx, mut_aa


def apply_single_mutation(seq: str, mut: str) -> Tuple[str, int]:
    """
    Apply single substitution like A23V to seq.
    Returns (mutated_seq, pos_0idx).
    """
    wt_aa, pos_1idx, mut_aa = parse_mutation(mut)
    pos_0idx = pos_1idx - 1

    if pos_0idx < 0 or pos_0idx >= len(seq):
        raise ValueError(f"Mutation {mut} position out of range for seq length {len(seq)}")

    if seq[pos_0idx] != wt_aa:
        raise ValueError(
            f"WT AA mismatch for {mut}: sequence has '{seq[pos_0idx]}' at position {pos_1idx}, expected '{wt_aa}'."
        )

    mutated = seq[:pos_0idx] + mut_aa + seq[pos_0idx + 1 :]
    return mutated, pos_0idx


def get_wt_sequence_from_df(sub: pd.DataFrame, mutant_col: str = "mutant") -> str:
    """
    Try to get WT sequence from known columns; otherwise reconstruct from mutants.
    If some positions are missing in the mutation coverage, fill with 'X'.
    """
    # 1) Try common WT sequence columns first
    candidate_cols = [
        "target_seq",
        "target_sequence",
        "wt_seq",
        "wt_sequence",
        "wildtype_sequence",
        "sequence",
        "protein_sequence",
        "aa_seq",
    ]

    for col in candidate_cols:
        if col in sub.columns:
            vals = sub[col].dropna().astype(str).unique().tolist()
            # If there's a single unique WT sequence, use it
            if len(vals) == 1 and len(vals[0]) > 10:
                return vals[0].strip().upper()

    # 2) Otherwise reconstruct from mutant strings (A23V style)
    pos_to_wt: Dict[int, str] = {}

    for m in sub[mutant_col].dropna().astype(str):
        m = m.strip().upper()
        if not m or len(m) < 3:
            continue
        pos_str = m[1:-1]
        if not pos_str.isdigit():
            continue
        wt_aa = m[0]
        pos = int(pos_str)

        if pos not in pos_to_wt:
            pos_to_wt[pos] = wt_aa
        else:
            if pos_to_wt[pos] != wt_aa:
                raise ValueError(
                    f"Inconsistent WT AA at position {pos}: {pos_to_wt[pos]} vs {wt_aa}. "
                    "This usually means mixed proteins/datasets."
                )

    if not pos_to_wt:
        raise ValueError("Could not reconstruct WT sequence: no valid single-sub mutants found.")

    max_pos = max(pos_to_wt.keys())

    # Build sequence 1..max_pos, filling missing with X
    missing = []
    seq = []
    for i in range(1, max_pos + 1):
        aa = pos_to_wt.get(i)
        if aa is None:
            missing.append(i)
            seq.append("X")
        else:
            seq.append(aa)

    if missing:
        print(
            "[WARN] Missing WT amino acids at positions (filled with 'X'): "
            + ", ".join(map(str, missing[:25]))
            + (" ..." if len(missing) > 25 else "")
        )

    return "".join(seq)



# ----------------------------
# ESM helpers
# ----------------------------
def load_esm_model(model_name: str, device: str) -> Tuple[torch.nn.Module, object]:
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model = model.to(device)
    return model, alphabet


@torch.no_grad()
def get_token_embeddings(
    model: torch.nn.Module,
    alphabet,
    sequence: str,
    device: str,
    repr_layer: Optional[int] = None,
) -> np.ndarray:
    """
    Returns token embeddings as (L, D) numpy array (excluding BOS/EOS).
    Uses last layer by default.
    """
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    if repr_layer is None:
        repr_layer = model.num_layers

    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = out["representations"][repr_layer]  # (B, T, D)

    # Strip BOS/EOS to get (L, D)
    reps = reps[0, 1 : len(sequence) + 1, :]
    return reps.detach().cpu().numpy()


def pooled_embedding(token_emb: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return token_emb.mean(axis=0)
    if mode == "sum":
        return token_emb.sum(axis=0)
    raise ValueError(f"Unknown pooling mode: {mode}")


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    cos_sim = float(np.dot(a, b) / denom)
    return float(1.0 - cos_sim)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def per_position_delta_l2(wt_tokens: np.ndarray, mut_tokens: np.ndarray) -> np.ndarray:
    delta = mut_tokens - wt_tokens
    return np.linalg.norm(delta, axis=1)


# ----------------------------
# ProteinGym loading
# ----------------------------
def load_proteingym_rows(
    parquet_path: Path,
    dms_ids: List[str],
    mutant_col: str = "mutant",
    dms_id_col: str = "DMS_id",
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if dms_id_col not in df.columns:
        raise KeyError(f"Expected column '{dms_id_col}' not found in parquet.")
    if mutant_col not in df.columns:
        raise KeyError(f"Expected column '{mutant_col}' not found in parquet.")

    sub = df[df[dms_id_col].astype(str).isin(dms_ids)].copy()

    # Strict single substitutions like A23V
    mask = sub[mutant_col].astype(str).str.match(r"^[A-Z][0-9]+[A-Z]$")
    sub = sub[mask].copy()

    if len(sub) == 0:
        raise ValueError(
            "No rows found after filtering to dms_ids + single substitutions.\n"
            f"dms_ids={dms_ids}\n"
            "Tip: run your GFP discovery script to confirm exact DMS_id strings."
        )

    return sub


# ----------------------------
# Main computation
# ----------------------------
def compute_deltas(
    wt_seq: str,
    mutations: List[str],
    model_name: str,
    pooling: str,
    device: str,
    repr_layer: Optional[int],
) -> List[DeltaResult]:
    model, alphabet = load_esm_model(model_name, device=device)

    wt_tok = get_token_embeddings(model, alphabet, wt_seq, device=device, repr_layer=repr_layer)
    wt_pool = pooled_embedding(wt_tok, mode=pooling)

    results: List[DeltaResult] = []

    for mut in mutations:
        mut_seq, pos0 = apply_single_mutation(wt_seq, mut)
        mut_tok = get_token_embeddings(model, alphabet, mut_seq, device=device, repr_layer=repr_layer)
        mut_pool = pooled_embedding(mut_tok, mode=pooling)

        cos_d = cosine_distance(wt_pool, mut_pool)
        l2_d = l2_distance(wt_pool, mut_pool)
        per_pos = per_position_delta_l2(wt_tok, mut_tok)

        results.append(
            DeltaResult(
                mutation=mut.upper(),
                mutant_position_1idx=pos0 + 1,
                global_cosine_distance=cos_d,
                global_l2_distance=l2_d,
                per_position_l2=per_pos.astype(float).tolist(),
            )
        )

    return results


def save_outputs(out_dir: Path, wt_seq: str, dms_ids: List[str], results: List[DeltaResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dms_ids": dms_ids,
        "sequence_length": len(wt_seq),
        "n_mutations": len(results),
        "results": [
            {
                "mutation": r.mutation,
                "mutant_position_1idx": r.mutant_position_1idx,
                "global_cosine_distance": r.global_cosine_distance,
                "global_l2_distance": r.global_l2_distance,
            }
            for r in results
        ],
    }
    (out_dir / "embedding_distance_summary.json").write_text(json.dumps(summary, indent=2))

    for r in results:
        (out_dir / f"per_position_delta_{r.mutation}.json").write_text(
            json.dumps(
                {
                    "mutation": r.mutation,
                    "mutant_position_1idx": r.mutant_position_1idx,
                    "per_position_l2": r.per_position_l2,
                },
                indent=2,
            )
        )

    # Also save WT sequence for reproducibility
    (out_dir / "wt_sequence.txt").write_text(wt_seq + "\n")

    print(f"Saved outputs to: {out_dir}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet_path",
        type=str,
        default=str(DEFAULT_PARQUET),
        help="Path to proteingym_dms_substitutions.parquet",
    )
    p.add_argument(
        "--dms_ids",
        type=str,
        nargs="+",
        required=True,
        help="One or more DMS_id values (e.g., GFP_AEQVI_Sarkisyan_2016)",
    )
    p.add_argument(
        "--mutations",
        type=str,
        nargs="+",
        required=True,
        help="Mutation strings like A23V S65T G150D (must match WT residues)",
    )
    p.add_argument("--out_dir", type=str, default="artifacts/embedding_deltas", help="Output directory")
    p.add_argument("--model_name", type=str, default="esm2_t12_35M_UR50D", help="ESM pretrained model name")
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "sum"], help="Pooling for global embedding")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--repr_layer", type=int, default=None, help="ESM layer to extract (default: last layer)")
    return p.parse_args()

def suggest_valid_mutations(sub: pd.DataFrame, n: int = 20) -> None:
    muts = (
        sub["mutant"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    print(f"\nExample valid mutations from parquet (showing {min(n, len(muts))}):")
    for m in muts[:n]:
        print("  ", m)


def main() -> None:
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    parquet_path = Path(args.parquet_path)

    # 1) Load selected ProteinGym rows (only for WT reconstruction)
    sub = load_proteingym_rows(parquet_path=parquet_path, dms_ids=args.dms_ids)

    suggest_valid_mutations(sub, n=30)


    # 2) Reconstruct WT sequence from mutant strings
    wt_seq = get_wt_sequence_from_df(sub, mutant_col="mutant")

    print("Reconstructed WT sequence length:", len(wt_seq))
    print("WT sequence (first 60 aa):", wt_seq[:60])

    # 3) Compute deltas for requested mutations
    results = compute_deltas(
        wt_seq=wt_seq,
        mutations=args.mutations,
        model_name=args.model_name,
        pooling=args.pooling,
        device=device,
        repr_layer=args.repr_layer,
    )

    # 4) Save outputs
    out_dir = Path(args.out_dir)
    save_outputs(out_dir=out_dir, wt_seq=wt_seq, dms_ids=args.dms_ids, results=results)

    # 5) Print quick table
    print("\nMutation results:")
    for r in results:
        print(
            f"{r.mutation:8s}  pos={r.mutant_position_1idx:4d}  "
            f"cos_dist={r.global_cosine_distance:.4f}  l2_dist={r.global_l2_distance:.4f}"
        )


if __name__ == "__main__":
    main()
