import numpy as np
import pandas as pd
from pathlib import Path
import torch
import esm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "proteingym_gfp_sarkisyan2016.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_seq_cols(df: pd.DataFrame) -> tuple[str, str]:
    # Try common ProteinGym names
    candidates = [
        ("target_seq", "mutated_sequence"),
        ("wt_sequence", "mut_sequence"),
        ("sequence", "mutated_sequence"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    raise KeyError(f"Could not find WT/mutant sequence columns. Columns are: {df.columns.tolist()}")

@torch.no_grad()
def embed_sequences(model, alphabet, seqs, device, batch_size=8):
    batch_converter = alphabet.get_batch_converter()
    embs = []

    model.eval()
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i+batch_size]
        data = [(f"s{k}", s) for k, s in enumerate(chunk)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)

        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        reps = out["representations"][model.num_layers]  # (B, T, C)

        # Mean pool over residues (exclude BOS/EOS)
        for b in range(reps.size(0)):
            # toks[b] includes BOS at 0; EOS may exist depending on alphabet
            # Use token count from seq length (+1 BOS)
            L = len(chunk[b])
            token_slice = reps[b, 1:1+L, :]  # residues only
            embs.append(token_slice.mean(dim=0).cpu().numpy())

    return np.vstack(embs)

def main():
    df = pd.read_parquet(IN_PATH)
    wt_col, mut_col = get_seq_cols(df)

    # Labels
    y = df["DMS_score_bin"].astype(int).to_numpy()

    # Sequences
    wt_seq = df[wt_col].iloc[0]
    mut_seqs = df[mut_col].tolist()

    # Load ESM2 (pick small-ish for speed)
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Embed WT once
    wt_emb = embed_sequences(model, alphabet, [wt_seq], device=device, batch_size=1)[0]

    # Embed mutants
    mut_embs = embed_sequences(model, alphabet, mut_seqs, device=device, batch_size=8)

    # Delta
    X = mut_embs - wt_emb[None, :]

    # Save
    np.save(OUT_DIR / "gfp_delta_X.npy", X)
    np.save(OUT_DIR / "gfp_y.npy", y)

    meta_cols = [c for c in ["mutant", "DMS_id", "DMS_score", "DMS_score_bin"] if c in df.columns]
    df[meta_cols].to_parquet(OUT_DIR / "gfp_meta.parquet", index=False)

    print("Saved X:", (OUT_DIR / "gfp_delta_X.npy"))
    print("Saved y:", (OUT_DIR / "gfp_y.npy"))
    print("X shape:", X.shape, "y shape:", y.shape)

if __name__ == "__main__":
    main()
