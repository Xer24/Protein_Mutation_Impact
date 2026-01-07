# src/viz/analyze_predictions.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def analyze_predictions(predictions_csv: Path, output_dir: Path):
    """
    Create comprehensive analysis plots from prediction CSV.
    """
    df = pd.read_csv(predictions_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse mutation positions
    df['pos'] = df['mutation'].str[1:-1].astype(int)
    df['from_aa'] = df['mutation'].str[0]
    df['to_aa'] = df['mutation'].str[-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probability distribution
    ax = axes[0, 0]
    ax.hist(df['probability_deleterious'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('P(deleterious)')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution')
    ax.axvline(df['probability_deleterious'].median(), color='red', 
               linestyle='--', label=f'Median: {df["probability_deleterious"].median():.3f}')
    ax.legend()
    
    # 2. Delta L2 vs Probability
    ax = axes[0, 1]
    scatter = ax.scatter(df['delta_l2'], df['probability_deleterious'], 
                        alpha=0.5, s=20)
    ax.set_xlabel('Delta L2 (embedding change)')
    ax.set_ylabel('P(deleterious)')
    ax.set_title('Embedding Change vs Deleteriousness')
    
    # 3. Position-wise mean probability
    ax = axes[1, 0]
    pos_stats = df.groupby('pos')['probability_deleterious'].mean().sort_index()
    ax.plot(pos_stats.index, pos_stats.values, linewidth=2)
    ax.set_xlabel('Position')
    ax.set_ylabel('Mean P(deleterious)')
    ax.set_title('Positional Sensitivity')
    ax.grid(alpha=0.3)
    
    # 4. Top 20 most deleterious
    ax = axes[1, 1]
    top20 = df.nlargest(20, 'probability_deleterious')
    ax.barh(range(len(top20)), top20['probability_deleterious'])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['mutation'], fontsize=8)
    ax.set_xlabel('P(deleterious)')
    ax.set_title('Top 20 Most Deleterious')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_dir / 'prediction_analysis.png'}")
    
    # Print summary stats
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total mutations: {len(df)}")
    print(f"Mean P(deleterious): {df['probability_deleterious'].mean():.3f}")
    print(f"Median P(deleterious): {df['probability_deleterious'].median():.3f}")
    print(f"Predicted deleterious: {(df['predicted_class'] == 1).sum()} ({(df['predicted_class'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"\nTop 5 most deleterious:")
    for i, row in df.nlargest(5, 'probability_deleterious').iterrows():
        print(f"  {row['mutation']}: {row['probability_deleterious']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions CSV")
    parser.add_argument("--output", type=str, default="artifacts/figures",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    analyze_predictions(Path(args.predictions), Path(args.output))