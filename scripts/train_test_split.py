import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_PATH = PROJECT_ROOT / "data" / "processed" / "gfp_delta_X.npy"
Y_PATH = PROJECT_ROOT / "data" / "processed" / "gfp_y.npy"

OUT_DIR = PROJECT_ROOT / "artifacts" / "train_test"

def main():
    # Load datapy scripts/03_train_test_split.py

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) != len(y):
        raise ValueError("X and y have different lengths — they must align row-wise")

    # Train/test split (STRATIFIED)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "X_test.npy", X_test)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_test.npy", y_test)

    # Sanity checks
    print("Train size:", len(y_train))
    print("Test size :", len(y_test))
    print("Positive rate (overall):", y.mean())
    print("Positive rate (train)  :", y_train.mean())
    print("Positive rate (test)   :", y_test.mean())

if __name__ == "__main__":
    main()
