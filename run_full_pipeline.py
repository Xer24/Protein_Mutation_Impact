"""
run_full_pipeline.py - Complete end-to-end ML pipeline for mutation prediction

Steps:
1. Build features (ESM2 embeddings + deltas)
2. Train/test split
3. Train model with SMOTE
4. Calibrate model
5. Validate performance
6. Generate all predictions
7. Visualize results
"""

import subprocess
import sys
from pathlib import Path
import time
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent

# Color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, total_steps, message):
    """Print a formatted step header."""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}STEP {step_num}/{total_steps}: {message}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"{Colors.OKBLUE}▶ {description}...{Colors.ENDC}")
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start
        print(f"{Colors.OKGREEN}✓ Completed in {elapsed:.1f}s{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}✗ Failed: {e}{Colors.ENDC}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if filepath.exists():
        print(f"{Colors.OKGREEN}✓ Found: {description}{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Missing: {description}{Colors.ENDC}")
        return False

def copy_files_if_needed():
    """Copy feature files from artifacts to data/processed if needed."""
    artifacts_X = PROJECT_ROOT / "artifacts" / "features" / "gfp_delta_X.npy"
    artifacts_y = PROJECT_ROOT / "artifacts" / "features" / "gfp_y.npy"
    
    data_X = PROJECT_ROOT / "data" / "processed" / "gfp_delta_X.npy"
    data_y = PROJECT_ROOT / "data" / "processed" / "gfp_y.npy"
    
    # Create data/processed directory if needed
    data_X.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy if artifacts files exist but data files don't
    if artifacts_X.exists() and not data_X.exists():
        print(f"{Colors.OKBLUE}📋 Copying features to data/processed/{Colors.ENDC}")
        shutil.copy(artifacts_X, data_X)
        shutil.copy(artifacts_y, data_y)
        print(f"{Colors.OKGREEN}✓ Files copied{Colors.ENDC}")

def main():
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("="*80)
    print("  PROTEIN MUTATION IMPACT PREDICTION - FULL PIPELINE")
    print("="*80)
    print(f"{Colors.ENDC}\n")
    
    total_steps = 8
    
    # ========================================================================
    # STEP 0: Optional - Clean artifacts
    # ========================================================================
    clean = input(f"{Colors.WARNING}Clean all artifacts and start fresh? (y/n): {Colors.ENDC}").lower()
    if clean == 'y':
        artifacts_dir = PROJECT_ROOT / "artifacts"
        data_processed = PROJECT_ROOT / "data" / "processed"
        
        if artifacts_dir.exists():
            print(f"{Colors.WARNING}🗑️  Deleting artifacts...{Colors.ENDC}")
            shutil.rmtree(artifacts_dir)
            artifacts_dir.mkdir()
            print(f"{Colors.OKGREEN}✓ Artifacts cleaned{Colors.ENDC}")
        
        # Also clean old feature files
        if data_processed.exists():
            for f in data_processed.glob("gfp_*.npy"):
                f.unlink()
            print(f"{Colors.OKGREEN}✓ Old features cleaned{Colors.ENDC}")
    
    # ========================================================================
    # STEP 1: Build Features
    # ========================================================================
    print_step(1, total_steps, "Building ESM2 Features (Delta Embeddings)")
    
    if not run_command(
        "python src/features/esm2_build_feature_matrix.py",
        "Generating ESM2 embeddings and computing deltas"
    ):
        print(f"{Colors.FAIL}Pipeline failed at feature building{Colors.ENDC}")
        sys.exit(1)
    
    # Copy files if they're in the wrong location
    copy_files_if_needed()
    
    # Verify outputs (check both locations)
    X_path = PROJECT_ROOT / "data" / "processed" / "gfp_delta_X.npy"
    y_path = PROJECT_ROOT / "data" / "processed" / "gfp_y.npy"
    
    if not (check_file_exists(X_path, "Feature matrix (X)") and 
            check_file_exists(y_path, "Labels (y)")):
        print(f"{Colors.FAIL}Feature building failed - missing outputs{Colors.ENDC}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Train/Test Split
    # ========================================================================
    print_step(2, total_steps, "Creating Train/Test Split")
    
    if not run_command(
        "python scripts/train_test_split.py",
        "Splitting data (80/20 stratified)"
    ):
        print(f"{Colors.FAIL}Pipeline failed at train/test split{Colors.ENDC}")
        sys.exit(1)
    
    # Verify outputs
    split_dir = PROJECT_ROOT / "artifacts" / "train_test"
    required_files = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    
    for fname in required_files:
        if not check_file_exists(split_dir / fname, fname):
            print(f"{Colors.FAIL}Split failed - missing {fname}{Colors.ENDC}")
            sys.exit(1)
    
    # ========================================================================
    # STEP 3: Train Model with SMOTE
    # ========================================================================
    print_step(3, total_steps, "Training Model with SMOTE + Threshold Optimization")
    
    if not run_command(
        "python src/models/train_baseline_from_computed_x.py",
        "Training Random Forest with class balancing"
    ):
        print(f"{Colors.FAIL}Pipeline failed at model training{Colors.ENDC}")
        sys.exit(1)
    
    # Verify model
    model_path = PROJECT_ROOT / "artifacts" / "models" / "rf_smote_bundle.joblib"
    if not check_file_exists(model_path, "Trained model"):
        print(f"{Colors.FAIL}Training failed - no model saved{Colors.ENDC}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: Calibrate Model
    # ========================================================================
    print_step(4, total_steps, "Calibrating Probabilities")
    
    if not run_command(
        "python scripts/calibrate.py",
        "Calibrating model and re-optimizing threshold"
    ):
        print(f"{Colors.FAIL}Pipeline failed at calibration{Colors.ENDC}")
        sys.exit(1)
    
    # Verify calibrated model
    cal_model_path = PROJECT_ROOT / "artifacts" / "models" / "rf_smote_calibrated_bundle.joblib"
    if not check_file_exists(cal_model_path, "Calibrated model"):
        print(f"{Colors.FAIL}Calibration failed - no calibrated model{Colors.ENDC}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 5: Validate Model
    # ========================================================================
    print_step(5, total_steps, "Validating Model Performance")
    
    if not run_command(
        "python scripts/check.py",
        "Analyzing performance at different thresholds"
    ):
        print(f"{Colors.WARNING}⚠ Validation script had issues (non-critical){Colors.ENDC}")
    
    # ========================================================================
    # STEP 6: Generate Predictions for All Mutations
    # ========================================================================
    print_step(6, total_steps, "Generating Predictions for All Mutations")
    
    predictions_path = PROJECT_ROOT / "artifacts" / "predictions" / "all_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not run_command(
        f"python src/inference/run_prediction.py --all --output {predictions_path}",
        "Running inference on all mutations"
    ):
        print(f"{Colors.FAIL}Pipeline failed at prediction generation{Colors.ENDC}")
        sys.exit(1)
    
    if not check_file_exists(predictions_path, "Predictions CSV"):
        print(f"{Colors.FAIL}Predictions failed - no output file{Colors.ENDC}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 7: Visualize Results
    # ========================================================================
    print_step(7, total_steps, "Creating Visualizations")
    
    if not run_command(
        f"python src/viz/analyze_predictions.py --predictions {predictions_path}",
        "Generating analysis plots"
    ):
        print(f"{Colors.WARNING}⚠ Visualization script had issues (non-critical){Colors.ENDC}")
    
    # ========================================================================
    # STEP 8: Generate Summary Report
    # ========================================================================
    print_step(8, total_steps, "Generating Summary Report")
    
    if not run_command(
        "python scripts/generate_report.py",  # ← Fixed typo (was "reporty")
        "Creating final summary report"
    ):
        print(f"{Colors.WARNING}⚠ Report generation had issues (non-critical){Colors.ENDC}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}")
    print("="*80)
    print("  ✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"{Colors.ENDC}\n")
    
    print(f"{Colors.OKCYAN}📁 Key Outputs:{Colors.ENDC}")
    print(f"  • Model: artifacts/models/rf_smote_calibrated_bundle.joblib")
    print(f"  • Predictions: {predictions_path}")
    print(f"  • Figures: artifacts/figures/")
    print(f"  • Report: artifacts/summary_report.txt")
    
    print(f"\n{Colors.OKCYAN}📊 Next Steps:{Colors.ENDC}")
    print(f"  • Review plots in artifacts/figures/")
    print(f"  • Check predictions CSV for specific mutations")
    print(f"  • Read summary report for model performance")
    
    print(f"\n{Colors.OKGREEN}✨ All done!{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠ Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}✗ Pipeline failed with error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)