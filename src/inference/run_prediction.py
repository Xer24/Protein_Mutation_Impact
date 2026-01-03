from pathlib import Path
from src.inference.predict import predict_mutation_effects
from src.features.esm2_embedding import ESM2Config

def main():
    wt = "MRKLSDELLIESYFKATEMNLNRDFIELIENEIKRRSLGHIISV"
    muts = ["A16C", "A16D", "A16V"]

    cfg = ESM2Config(
        model_name="esm2_t12_35M_UR50D",
        cache_path=Path("artifacts/esm2_cache_t12_mean.joblib"),
    )

    out = predict_mutation_effects(
        wt_seq=wt,
        mutations=muts,
        model_path="artifacts/mlp_gfp.joblib",
        cfg=cfg,
    )

    for m, p in zip(out["mutations"], out["prob_deleterious"]):
        print(m, round(float(p), 3))

if __name__ == "__main__":
    main()
