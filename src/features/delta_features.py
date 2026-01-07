# src/viz/plot_predictions.py

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_prediction_distribution(predictions_csv: Path, output_dir: Path):
    """Plot distribution of predictions from CSV output."""
    df = pd.read_csv(predictions_csv)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['probability_deleterious'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('P(deleterious)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14)
    plt.axvline(0.5, color='red', linestyle='--', label='Default threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution.png', dpi=150)
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'prediction_distribution.png'}")