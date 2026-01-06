from pathlib import Path
import random
import pandas as pd

from src.inference.predict import predict_mutation_effects
from src.features.esm2_embedding import ESM2Config
from src.mutations.parse_mutation import scan_all_positions  # adjust if needed


def save_predictions(out, path: Path):
    df = pd.DataFrame({
        "mutation": out["mutations"],
        "prob_deleterious": out["prob_deleterious"],
        "delta_l2": out["delta_l2"],
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved predictions to {path.resolve()}")


def main():
    wt = "MRKLSDELLIESYFKATEMNLNRDFIELIENEIKRRSLGHIISV"

    all_muts = scan_all_positions(wt)
    print("Total possible single mutations:", len(all_muts))

    K = 836
    if K > len(all_muts):
        raise ValueError("K exceeds total number of possible single mutations")

    random.seed(42)
    muts = random.sample(all_muts, k=K)

    cfg = ESM2Config(
        model_name="esm2_t12_35M_UR50D",
        cache_path=Path("artifacts/esm2_cache_t12_mean.joblib"),
        batch_size=32,
    )

    out = predict_mutation_effects(
        wt_seq=wt,
        mutations=muts,
        model_path="artifacts/mlp_gfp.joblib",
        cfg=cfg,
    )

    # ✅ SAVE HERE (this is what was missing)
    save_predictions(out, Path("artifacts/predictions/predictions_k836.parquet"))

    rows = list(zip(out["mutations"], out["prob_deleterious"], out["delta_l2"]))
    rows.sort(key=lambda x: x[1], reverse=True)

    print("Top predicted deleterious:")
    for m, p, d in rows[:20]:
        print(m, round(float(p), 3), round(float(d), 3))


if __name__ == "__main__":
    main()
