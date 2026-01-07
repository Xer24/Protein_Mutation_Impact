"""
analyze_predictions.py - Comprehensive visualization of prediction results
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def analyze_predictions(predictions_csv: Path, output_dir: Path):
    """
    Create comprehensive analysis plots from prediction CSV.
    """
    print(f"📊 Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse mutation positions and amino acids
    df['pos'] = df['mutation'].str[1:-1].astype(int)
    df['from_aa'] = df['mutation'].str[0]
    df['to_aa'] = df['mutation'].str[-1]
    
    print(f"   Total mutations: {len(df)}")
    print(f"   Predicted deleterious: {(df['predicted_class'] == 1).sum()} ({(df['predicted_class'] == 1).sum()/len(df)*100:.1f}%)")
    
    # ========================================================================
    # Figure 1: 4-Panel Overview
    # ========================================================================
    print("   Creating overview figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Probability distribution
    ax = axes[0, 0]
    ax.hist(df['probability_deleterious'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(df['probability_deleterious'].median(), color='red', linestyle='--', 
               linewidth=2, label=f'Median: {df["probability_deleterious"].median():.3f}')
    ax.axvline(df['probability_deleterious'].mean(), color='orange', linestyle='--', 
               linewidth=2, label=f'Mean: {df["probability_deleterious"].mean():.3f}')
    ax.set_xlabel('P(deleterious)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel 2: Delta L2 vs Probability scatter
    ax = axes[0, 1]
    scatter = ax.scatter(df['delta_l2'], df['probability_deleterious'], 
                        c=df['predicted_class'], cmap='RdYlGn_r', alpha=0.6, s=30)
    ax.set_xlabel('Delta L2 (Embedding Change)', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(deleterious)', fontsize=12, fontweight='bold')
    ax.set_title('Embedding Change vs Deleteriousness', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Predicted Class', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel 3: Position-wise mean probability
    ax = axes[1, 0]
    pos_stats = df.groupby('pos').agg({
        'probability_deleterious': ['mean', 'std', 'count']
    }).reset_index()
    pos_stats.columns = ['pos', 'mean_prob', 'std_prob', 'count']
    pos_stats = pos_stats[pos_stats['count'] >= 3]  # At least 3 mutations per position
    
    ax.plot(pos_stats['pos'], pos_stats['mean_prob'], linewidth=2, color='darkblue')
    ax.fill_between(pos_stats['pos'], 
                     pos_stats['mean_prob'] - pos_stats['std_prob'],
                     pos_stats['mean_prob'] + pos_stats['std_prob'],
                     alpha=0.3, color='lightblue')
    ax.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean P(deleterious)', fontsize=12, fontweight='bold')
    ax.set_title('Positional Sensitivity (≥3 mutations per position)', fontsize=14, fontweight='bold')
    ax.axhline(df['probability_deleterious'].mean(), color='red', linestyle='--', 
               alpha=0.5, label='Overall mean')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel 4: Top 20 most deleterious
    ax = axes[1, 1]
    top20 = df.nlargest(20, 'probability_deleterious')
    colors = plt.cm.Reds(top20['probability_deleterious'])
    ax.barh(range(len(top20)), top20['probability_deleterious'], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['mutation'], fontsize=9)
    ax.set_xlabel('P(deleterious)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Deleterious Mutations', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    overview_path = output_dir / 'prediction_overview.png'
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {overview_path}")
    
    # ========================================================================
    # Figure 2: Amino Acid Substitution Heatmap
    # ========================================================================
    print("   Creating substitution heatmap...")
    
    # Create substitution matrix
    aa_order = list('ACDEFGHIKLMNPQRSTVWY')
    sub_matrix = pd.DataFrame(0.0, index=aa_order, columns=aa_order)
    sub_counts = pd.DataFrame(0, index=aa_order, columns=aa_order)
    
    for _, row in df.iterrows():
        from_aa = row['from_aa']
        to_aa = row['to_aa']
        if from_aa in aa_order and to_aa in aa_order:
            sub_matrix.loc[from_aa, to_aa] += row['probability_deleterious']
            sub_counts.loc[from_aa, to_aa] += 1
    
    # Average probabilities
    sub_matrix = sub_matrix / sub_counts.replace(0, np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_matrix, annot=False, cmap='RdYlGn_r', center=0.5, 
                vmin=0, vmax=1, cbar_kws={'label': 'Mean P(deleterious)'}, ax=ax)
    ax.set_xlabel('To Amino Acid', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Amino Acid (WT)', fontsize=12, fontweight='bold')
    ax.set_title('Amino Acid Substitution Impact Matrix', fontsize=14, fontweight='bold')
    
    heatmap_path = output_dir / 'substitution_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {heatmap_path}")
    
    # ========================================================================
    # Figure 3: Position Heatmap
    # ========================================================================
    print("   Creating position heatmap...")
    
    # Get top 50 most sensitive positions
    pos_sensitivity = df.groupby('pos')['probability_deleterious'].mean().sort_values(ascending=False).head(50)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create pivot table for heatmap
    pivot_data = df[df['pos'].isin(pos_sensitivity.index)].pivot_table(
        values='probability_deleterious',
        index='to_aa',
        columns='pos',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, cmap='RdYlGn_r', center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Mean P(deleterious)'}, ax=ax)
    ax.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mutant Amino Acid', fontsize=12, fontweight='bold')
    ax.set_title('Top 50 Most Sensitive Positions', fontsize=14, fontweight='bold')
    
    pos_heatmap_path = output_dir / 'position_heatmap.png'
    plt.savefig(pos_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {pos_heatmap_path}")
    
    # ========================================================================
    # Print Summary Statistics
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total mutations analyzed: {len(df)}")
    print(f"Mean P(deleterious): {df['probability_deleterious'].mean():.3f}")
    print(f"Median P(deleterious): {df['probability_deleterious'].median():.3f}")
    print(f"Std P(deleterious): {df['probability_deleterious'].std():.3f}")
    print(f"\nPredicted deleterious: {(df['predicted_class'] == 1).sum()} ({(df['predicted_class'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"Predicted tolerated: {(df['predicted_class'] == 0).sum()} ({(df['predicted_class'] == 0).sum()/len(df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("TOP 10 MOST DELETERIOUS MUTATIONS")
    print(f"{'='*80}")
    for i, row in df.nlargest(10, 'probability_deleterious').iterrows():
        print(f"  {i+1:2d}. {row['mutation']:8s}  P={row['probability_deleterious']:.4f}  ΔL2={row['delta_l2']:.4f}")
    
    print(f"\n{'='*80}")
    print("TOP 10 MOST TOLERATED MUTATIONS")
    print(f"{'='*80}")
    for i, row in df.nsmallest(10, 'probability_deleterious').iterrows():
        print(f"  {i+1:2d}. {row['mutation']:8s}  P={row['probability_deleterious']:.4f}  ΔL2={row['delta_l2']:.4f}")
    
    print(f"\n{'='*80}")
    print("TOP 10 MOST SENSITIVE POSITIONS")
    print(f"{'='*80}")
    pos_stats = df.groupby('pos').agg({
        'probability_deleterious': ['mean', 'std', 'count']
    }).reset_index()
    pos_stats.columns = ['pos', 'mean_prob', 'std_prob', 'count']
    pos_stats = pos_stats[pos_stats['count'] >= 5].sort_values('mean_prob', ascending=False)
    
    for i, row in pos_stats.head(10).iterrows():
        print(f"  Position {row['pos']:3.0f}: Mean P={row['mean_prob']:.3f} (±{row['std_prob']:.3f}, n={row['count']:.0f})")
    
    print(f"\n✓ All visualizations saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize mutation predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/figures",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    analyze_predictions(
        Path(args.predictions),
        Path(args.output)
    )