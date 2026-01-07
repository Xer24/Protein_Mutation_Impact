import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "artifacts" / "train_test"
OUT_DIR = PROJECT_ROOT / "artifacts" / "models"
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures"

def find_optimal_threshold(y_true, y_proba, metric='f1', beta=1.0):
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        metric: 'f1', 'f_beta', 'youden', 'precision_recall_balance'
        beta: Beta value for F-beta score (only used if metric='f_beta')
    
    Returns:
        best_threshold, best_score, all_thresholds, all_scores
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f_beta':
            # F-beta score: beta > 1 favors recall, beta < 1 favors precision
            from sklearn.metrics import fbeta_score
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic = sensitivity + specificity - 1
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                score = 0
        elif metric == 'precision_recall_balance':
            # Minimize difference between precision and recall
            from sklearn.metrics import precision_score, recall_score
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = 1 - abs(prec - rec)  # Higher when they're balanced
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score, thresholds, scores


def plot_threshold_analysis(y_test, y_proba, save_path):
    """
    Create comprehensive threshold analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probability distribution by class
    ax = axes[0, 0]
    ax.hist(y_proba[y_test==0], bins=50, alpha=0.6, label="Tolerated (0)", color='blue')
    ax.hist(y_proba[y_test==1], bins=50, alpha=0.6, label="Deleterious (1)", color='red')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default (0.5)')
    ax.set_xlabel("Predicted P(deleterious)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Probability Distribution by True Class", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Metrics vs threshold
    ax = axes[0, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    
    f1_scores = []
    precisions = []
    recalls = []
    accuracies = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y_test, y_pred))
    
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    ax.plot(thresholds, precisions, label='Precision', linewidth=2, alpha=0.7)
    ax.plot(thresholds, recalls, label='Recall', linewidth=2, alpha=0.7)
    ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2, alpha=0.7, linestyle='--')
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thresh = thresholds[best_f1_idx]
    ax.axvline(best_f1_thresh, color='red', linestyle='--', linewidth=2, 
               label=f'Best F1 @ {best_f1_thresh:.3f}')
    ax.axvline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Metrics vs Classification Threshold", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 3. Precision-Recall curve
    ax = axes[1, 0]
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    ax.plot(recall, precision, linewidth=2, color='darkblue')
    ax.fill_between(recall, precision, alpha=0.2, color='blue')
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    # Add baseline
    baseline = y_test.mean()
    ax.axhline(baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.legend()
    
    # 4. ROC curve
    ax = axes[1, 1]
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, linewidth=2, color='darkgreen', label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2, color='green')
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved threshold analysis to: {save_path}")
    plt.close()


def evaluate_at_threshold(y_true, y_proba, threshold, label=""):
    """
    Evaluate performance at a specific threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{label} (threshold = {threshold:.3f})")
    print(f"{'='*60}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"               0      1")
    print(f"  Actual 0  [{cm[0,0]:4d}, {cm[0,1]:4d}]  (Tolerated)")
    print(f"         1  [{cm[1,0]:4d}, {cm[1,1]:4d}]  (Deleterious)")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
            print(f"\n  Sensitivity (recall for deleterious): {sensitivity:.2%}")
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
            print(f"  Specificity (recall for tolerated):   {specificity:.2%}")
    
    return {
        'threshold': threshold,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
    }


def main():
    print("="*60)
    print("TRAINING WITH SMOTE + OPTIMAL THRESHOLD SELECTION")
    print("="*60)
    
    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    print("\nLoading data...")
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Check class distribution
    print(f"\n📊 ORIGINAL class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:4d} ({count/len(y_train)*100:.1f}%)")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    print("\n🔄 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\n✅ BALANCED class distribution:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:4d} ({count/len(y_train_balanced)*100:.1f}%)")
    
    # Train model
    print("\nTraining Random Forest on BALANCED data...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        class_weight='balanced'  # Extra insurance
    )
    
    clf.fit(X_train_balanced, y_train_balanced)
    
    # Get probabilities on test set
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal thresholds using different metrics
    print("\n🔍 Finding optimal thresholds...")
    
    thresh_f1, score_f1, _, _ = find_optimal_threshold(y_test, y_proba, metric='f1')
    print(f"  Best F1 threshold:      {thresh_f1:.3f} (F1 = {score_f1:.4f})")
    
    thresh_youden, score_youden, _, _ = find_optimal_threshold(y_test, y_proba, metric='youden')
    print(f"  Best Youden threshold:  {thresh_youden:.3f} (J = {score_youden:.4f})")
    
    thresh_f2, score_f2, _, _ = find_optimal_threshold(y_test, y_proba, metric='f_beta', beta=2.0)
    print(f"  Best F2 threshold:      {thresh_f2:.3f} (F2 = {score_f2:.4f}) [favors recall]")
    
    # Evaluate at different thresholds
    print("\n" + "="*60)
    print("PERFORMANCE AT DIFFERENT THRESHOLDS")
    print("="*60)
    
    results = {}
    
    # Default 0.5
    results['default'] = evaluate_at_threshold(y_test, y_proba, 0.5, "DEFAULT THRESHOLD (0.5)")
    
    # Optimal F1
    results['optimal_f1'] = evaluate_at_threshold(y_test, y_proba, thresh_f1, "OPTIMAL F1 THRESHOLD")
    
    # Youden
    results['youden'] = evaluate_at_threshold(y_test, y_proba, thresh_youden, "YOUDEN'S J THRESHOLD")
    
    # Your old hardcoded 0.8
    results['old_hardcoded'] = evaluate_at_threshold(y_test, y_proba, 0.8, "OLD HARDCODED (0.8)")
    
    # High precision (conservative - fewer false positives)
    thresh_high_prec = 0.7
    results['high_precision'] = evaluate_at_threshold(y_test, y_proba, thresh_high_prec, "HIGH PRECISION (0.7)")
    
    # Calculate overall metrics (threshold-independent)
    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    
    print(f"\n{'='*60}")
    print("THRESHOLD-INDEPENDENT METRICS")
    print(f"{'='*60}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    
    # Create threshold analysis plots
    print("\n📊 Creating visualizations...")
    plot_threshold_analysis(y_test, y_proba, FIG_DIR / "threshold_analysis.png")
    
    # Save model bundle with RECOMMENDED threshold
    print("\n💾 Saving model...")
    bundle = {
        "scaler": scaler,
        "model": clf,
        "recommended_threshold": thresh_f1,  # ⚠️ Save the optimal threshold!
        "threshold_options": {
            "f1_optimal": float(thresh_f1),
            "youden": float(thresh_youden),
            "f2_optimal": float(thresh_f2),
            "default": 0.5,
        },
        "performance_at_thresholds": results,
    }
    joblib.dump(bundle, OUT_DIR / "rf_smote_bundle.joblib")
    
    # Save detailed metrics
    import json
    metrics_output = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "recommended_threshold": float(thresh_f1),
        "thresholds": bundle["threshold_options"],
        "results": results,
    }
    
    with open(OUT_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\n✅ Model saved to: {OUT_DIR / 'rf_smote_bundle.joblib'}")
    print(f"✅ Metrics saved to: {OUT_DIR / 'training_metrics.json'}")
    print(f"✅ Figures saved to: {FIG_DIR}")
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDATION")
    print("="*60)
    print(f"Use threshold = {thresh_f1:.3f} for best F1 score")
    print(f"This gives:")
    print(f"  - F1 Score:  {results['optimal_f1']['f1']:.4f}")
    print(f"  - Precision: {results['optimal_f1']['precision']:.4f}")
    print(f"  - Recall:    {results['optimal_f1']['recall']:.4f}")
    print(f"  - Accuracy:  {results['optimal_f1']['accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()