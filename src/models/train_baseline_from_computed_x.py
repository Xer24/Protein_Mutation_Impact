import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
X_PATH = PROJECT_ROOT / "data" / "processed" / "gfp_delta_X.npy"
Y_PATH = PROJECT_ROOT / "data" / "processed" / "gfp_y.npy"
ART_DIR = PROJECT_ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # For GFP-only, a simple split is okay (single protein).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            max_iter=200,
            early_stopping=True,
            random_state=42,
        )),
    ])

    clf.fit(X_train, y_train)

    p = clf.predict_proba(X_test)[:, 1]
    yhat = (p >= 0.5).astype(int)

    metrics = {
        "auroc": float(roc_auc_score(y_test, p)),
        "auprc": float(average_precision_score(y_test, p)),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pos_rate_test": float(y_test.mean()),
    }

    joblib.dump(clf, ART_DIR / "mlp_gfp.joblib")
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved model:", ART_DIR / "mlp_gfp.joblib")
    print("Saved metrics:", ART_DIR / "metrics.json")
    print(metrics)
    
    plt.hist(p[y_test==0], bins=40, alpha=0.5, label="non-deleterious")
    plt.hist(p[y_test==1], bins=40, alpha=0.5, label="deleterious")
    plt.legend()
    plt.xlabel("Predicted P(deleterious)")
    plt.ylabel("Count")
    plt.show()

    baseline_rate = y_test.mean()
    print("Baseline positive rate:", baseline_rate)


    print("Fraction class 1 when p > 0.9:", (y_test[p > 0.9] == 1).mean())
    print("Fraction class 0 when p < 0.1:", (y_test[p < 0.1] == 0).mean())


"""p = predicted probability P(deleterious = 1)
x-axis = model confidence (0 → 1)
y-axis = number of mutations
blue (class 0) = mutations labeled non-deleterious
orange (class 1) = mutations labeled deleterious
"""""




if __name__ == "__main__":
    main()
