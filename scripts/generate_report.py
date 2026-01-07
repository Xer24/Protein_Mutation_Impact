"""
generate_report.py - Generate text summary report
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_report():
    """Generate comprehensive text report."""
    
    report_path = PROJECT_ROOT / "artifacts" / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("  PROTEIN MUTATION IMPACT PREDICTION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Model info
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        
        metrics_path = PROJECT_ROOT / "artifacts" / "models" / "calibration_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as mf:
                metrics = json.load(mf)
            
            f.write(f"Model Type: Random Forest with SMOTE balancing\n")
            f.write(f"Calibration Method: {metrics.get('method', 'N/A')}\n")
            f.write(f"Recommended Threshold: {metrics.get('calibrated_threshold', 'N/A'):.3f}\n")
            f.write(f"\nPerformance Metrics:\n")
            f.write(f"  AUROC (calibrated): {metrics.get('auroc_cal', 0):.4f}\n")
            f.write(f"  AUPRC (calibrated): {metrics.get('auprc_cal', 0):.4f}\n")
            f.write(f"  Brier Score (calibrated): {metrics.get('brier_cal', 0):.4f}\n")
        
        # Predictions summary
        f.write("\n" + "="*80 + "\n")
        f.write("PREDICTIONS SUMMARY\n")
        f.write("-"*80 + "\n")
        
        pred_path = PROJECT_ROOT / "artifacts" / "predictions" / "all_predictions.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            
            f.write(f"Total mutations analyzed: {len(df)}\n")
            f.write(f"Mean P(deleterious): {df['probability_deleterious'].mean():.3f}\n")
            f.write(f"Median P(deleterious): {df['probability_deleterious'].median():.3f}\n")
            f.write(f"\nPredicted deleterious: {(df['predicted_class'] == 1).sum()} ({(df['predicted_class'] == 1).sum()/len(df)*100:.1f}%)\n")
            f.write(f"Predicted tolerated: {(df['predicted_class'] == 0).sum()} ({(df['predicted_class'] == 0).sum()/len(df)*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated successfully!\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved to: {report_path}")


if __name__ == "__main__":
    generate_report()