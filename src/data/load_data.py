import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "raw" / "proteingym_dms_substitutions.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "proteingym_gfp_sarkisyan2016.parquet"

GFP_ID = "GFP_AEQVI_Sarkisyan_2016"

def main() -> None:
    print("Loading:", IN_PATH)
    df = pd.read_parquet(IN_PATH)
    print("Raw shape:", df.shape)

    # Filter to GFP assay
    gfp = df[df["DMS_id"] == GFP_ID].copy()
    print("GFP shape (raw):", gfp.shape)

    # Keep only single substitutions like A23V (ProteinGym usually uses 'mutant' column)
    if "mutant" in gfp.columns:
        gfp = gfp[gfp["mutant"].astype(str).str.match(r"^[A-Z][0-9]+[A-Z]$")]
    else:
        raise KeyError("Expected column 'mutant' not found. Print columns and adjust.")

    # Ensure binary label exists and is int
    if "DMS_score_bin" not in gfp.columns:
        raise KeyError("Expected column 'DMS_score_bin' not found. Print columns and adjust.")

    gfp = gfp.dropna(subset=["DMS_score_bin"]).copy()
    gfp["DMS_score_bin"] = gfp["DMS_score_bin"].astype(int)

    # Quick sanity checks
    print("GFP shape (clean):", gfp.shape)
    print("Label balance:")
    print(gfp["DMS_score_bin"].value_counts())
    print("Label balance (%):")
    print(gfp["DMS_score_bin"].value_counts(normalize=True).round(4))

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gfp.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
#gfp = pd.read_parquet("data/processed/proteingym_gfp_sarkisyan2016.parquet")