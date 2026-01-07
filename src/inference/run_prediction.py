"""
run_prediction.py - Easy interface for making predictions
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_mutation_effects

DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "rf_smote_calibrated_bundle.joblib"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "proteingym_gfp_sarkisyan2016.parquet"


def load_wt_from_parquet(parquet_path: Path, wt_col: str = "target_seq") -> str:
    """
    Load wild-type sequence from parquet file.
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        # Try common WT sequence column names
        possible_cols = [wt_col, "target_sequence", "wt_seq", "wt_sequence", "sequence"]
        
        for col in possible_cols:
            if col in df.columns:
                wt_seq = df[col].iloc[0]
                if isinstance(wt_seq, str) and len(wt_seq) > 10:
                    print(f"✅ Loaded WT sequence from '{col}' column (length: {len(wt_seq)})")
                    return wt_seq
        
        raise ValueError(f"Could not find WT sequence column. Available columns: {df.columns.tolist()}")
    
    except Exception as e:
        raise ValueError(f"Failed to load WT sequence from {parquet_path}: {e}")


def load_mutations_from_parquet(parquet_path: Path, mutation_col: str = "mutant") -> list:
    """
    Load all mutations from parquet file.
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        # Try common mutation column names
        possible_cols = [mutation_col, "mutation", "mutant", "mutations", "variant"]
        
        for col in possible_cols:
            if col in df.columns:
                mutations = df[col].dropna().astype(str).unique().tolist()
                # Filter to single substitutions only (A23V format)
                mutations = [m for m in mutations if pd.notna(m) and len(m) >= 3]
                print(f"✅ Loaded {len(mutations)} mutations from '{col}' column")
                return mutations
        
        raise ValueError(f"Could not find mutation column. Available columns: {df.columns.tolist()}")
    
    except Exception as e:
        raise ValueError(f"Failed to load mutations from {parquet_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict mutation effects on protein function",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict ALL mutations from data file
  python src/inference/run_prediction.py --all --output all_predictions.csv
  
  # Predict specific mutations (auto-load WT)
  python src/inference/run_prediction.py --mutations A206T S65T Y66H
  
  # Provide WT sequence manually
  python src/inference/run_prediction.py --wt_seq "MSKGEEL..." --mutations A206T S65T
  
  # Use custom data file
  python src/inference/run_prediction.py --data data/my_protein.parquet --all
  
  # Use custom threshold
  python src/inference/run_prediction.py --all --threshold 0.4 --output results.csv
        """
    )
    
    # === WT SEQUENCE ===
    parser.add_argument(
        "--wt_seq", 
        type=str, 
        default=None,
        help="Wild-type protein sequence (if not provided, will load from --data)"
    )
    
    # === MUTATIONS ===
    mutation_group = parser.add_mutually_exclusive_group(required=True)
    mutation_group.add_argument(
        "--mutations", 
        type=str, 
        nargs="+",
        help="Specific mutations to test (e.g., A206T S65T Y66H)"
    )
    mutation_group.add_argument(
        "--all",
        action="store_true",
        help="Predict ALL mutations from the data file"
    )
    
    # === DATA SOURCE ===
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help=f"Parquet file to load WT sequence and/or mutations from"
    )
    
    # === MODEL ===
    parser.add_argument(
        "--model", 
        type=str, 
        default=str(DEFAULT_MODEL),
        help="Path to model bundle"
    )
    
    # === THRESHOLD ===
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None,
        help="Custom threshold (default: use optimal from training)"
    )
    
    # === OUTPUT ===
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Save results to CSV (required when using --all)"
    )
    
    # === OPTIONS ===
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Process mutations in batches (for --all mode)"
    )
    
    args = parser.parse_args()
    
    # === VALIDATION ===
    if args.all and not args.output:
        print("❌ Error: --output is required when using --all")
        print("   (You don't want to print thousands of predictions to console!)")
        sys.exit(1)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # === GET WT SEQUENCE ===
    if args.wt_seq is None:
        if not args.quiet:
            print(f"📂 Loading WT sequence from: {data_path.name}")
        try:
            wt_seq = load_wt_from_parquet(data_path)
        except Exception as e:
            print(f"❌ Error loading WT sequence: {e}")
            print(f"\nPlease provide --wt_seq explicitly")
            sys.exit(1)
    else:
        wt_seq = args.wt_seq
        if not args.quiet:
            print(f"✅ Using provided WT sequence (length: {len(wt_seq)})")
    
    # === GET MUTATIONS ===
    if args.all:
        if not args.quiet:
            print(f"📂 Loading ALL mutations from: {data_path.name}")
        try:
            mutations = load_mutations_from_parquet(data_path)
        except Exception as e:
            print(f"❌ Error loading mutations: {e}")
            sys.exit(1)
    else:
        mutations = args.mutations
    
    if not args.quiet:
        print(f"\n🧬 Predicting mutation effects...")
        print(f"   WT sequence length: {len(wt_seq)}")
        print(f"   Mutations to test: {len(mutations)}")
        print(f"   Model: {Path(args.model).name}")
    
    # === MAKE PREDICTIONS ===
    # For large datasets, process in batches
    if len(mutations) > args.batch_size and not args.quiet:
        print(f"   Processing in batches of {args.batch_size}...")
    
    all_results = {
        'mutations': [],
        'mutant_seqs': [],
        'delta_l2': [],
        'prob_deleterious': [],
        'predicted_deleterious': [],
    }
    
    try:
        # Process in batches
        for i in range(0, len(mutations), args.batch_size):
            batch = mutations[i:i + args.batch_size]
            
            if not args.quiet and len(mutations) > args.batch_size:
                print(f"   Batch {i//args.batch_size + 1}/{(len(mutations)-1)//args.batch_size + 1}...", end='\r')
            
            results = predict_mutation_effects(
                wt_seq=wt_seq,
                mutations=batch,
                model_path=args.model,
                use_calibrated=True,
                use_optimal_threshold=True,
                custom_threshold=args.threshold,
            )
            
            # Accumulate results
            all_results['mutations'].extend(results['mutations'])
            all_results['mutant_seqs'].extend(results['mutant_seqs'])
            all_results['delta_l2'] = np.concatenate([all_results['delta_l2'], results['delta_l2']]) if len(all_results['delta_l2']) > 0 else results['delta_l2']
            all_results['prob_deleterious'] = np.concatenate([all_results['prob_deleterious'], results['prob_deleterious']]) if len(all_results['prob_deleterious']) > 0 else results['prob_deleterious']
            all_results['predicted_deleterious'] = np.concatenate([all_results['predicted_deleterious'], results['predicted_deleterious']]) if len(all_results['predicted_deleterious']) > 0 else results['predicted_deleterious']
        
        # Store metadata from last batch
        all_results['threshold_used'] = results['threshold_used']
        all_results['model_type'] = results['model_type']
        
        if not args.quiet and len(mutations) > args.batch_size:
            print()  # New line after progress
    
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # === DISPLAY RESULTS ===
    if not args.quiet and not args.all:
        # Only display to console for small number of mutations
        print(f"\n{'='*80}")
        print(f"RESULTS (using {all_results['model_type']} model, threshold={all_results['threshold_used']:.3f})")
        print(f"{'='*80}")
        print(f"{'Mutation':<12} {'P(del)':<10} {'Prediction':<15} {'Delta L2':<10}")
        print(f"{'-'*80}")
        
        for i, mut in enumerate(all_results['mutations']):
            prob = all_results['prob_deleterious'][i]
            pred = "DELETERIOUS" if all_results['predicted_deleterious'][i] == 1 else "TOLERATED"
            delta = all_results['delta_l2'][i]
            print(f"{mut:<12} {prob:>6.4f}     {pred:<15} {delta:>8.4f}")
    
    # === SAVE TO CSV ===
    if args.output:
        df = pd.DataFrame({
            'mutation': all_results['mutations'],
            'probability_deleterious': all_results['prob_deleterious'],
            'predicted_class': all_results['predicted_deleterious'],
            'prediction_label': ['DELETERIOUS' if p == 1 else 'TOLERATED' 
                                for p in all_results['predicted_deleterious']],
            'delta_l2': all_results['delta_l2'],
            'mutant_sequence': all_results['mutant_seqs'],
        })
        
        # Sort by probability (most deleterious first)
        df = df.sort_values('probability_deleterious', ascending=False)
        
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        
        if not args.quiet:
            print(f"\n✅ Saved {len(df)} predictions to: {output_path}")
    
    # === SUMMARY ===
    if not args.quiet:
        n_del = sum(all_results['predicted_deleterious'])
        n_tol = len(all_results['predicted_deleterious']) - n_del
        
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY")
        print(f"{'='*80}")
        print(f"Total mutations:  {len(all_results['mutations'])}")
        print(f"Deleterious:      {n_del} ({n_del/len(all_results['mutations'])*100:.1f}%)")
        print(f"Tolerated:        {n_tol} ({n_tol/len(all_results['mutations'])*100:.1f}%)")
        print(f"\nThreshold used:   {all_results['threshold_used']:.3f}")
        print(f"Model type:       {all_results['model_type']}")
        
        # Show top 5 most and least deleterious
        if args.output and len(all_results['mutations']) > 10:
            print(f"\n🔴 Top 5 most deleterious:")
            for i in range(min(5, len(df))):
                mut = df.iloc[i]['mutation']
                prob = df.iloc[i]['probability_deleterious']
                print(f"   {i+1}. {mut}: {prob:.4f}")
            
            print(f"\n🟢 Top 5 most tolerated:")
            for i in range(max(0, len(df)-5), len(df)):
                mut = df.iloc[i]['mutation']
                prob = df.iloc[i]['probability_deleterious']
                print(f"   {len(df)-i}. {mut}: {prob:.4f}")


if __name__ == "__main__":
    main()