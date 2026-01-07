# src/utils/sequence_utils.py

import pandas as pd

def reconstruct_wt_sequence(
    df: pd.DataFrame,
    mutation_col: str = "mutant",
) -> str:
    """
    Reconstruct WT sequence from single-substitution mutations.
    Useful for validation and when WT sequence isn't directly available.
    """
    pos_to_aa = {}

    for m in df[mutation_col].dropna().astype(str):
        if not m or len(m) < 3:
            continue
        
        wt, mut = m[0], m[-1]
        pos_str = m[1:-1]
        
        if not pos_str.isdigit():
            continue
        
        pos = int(pos_str)
        
        if pos not in pos_to_aa:
            pos_to_aa[pos] = wt
        elif pos_to_aa[pos] != wt:
            raise ValueError(
                f"Inconsistent WT at position {pos}: {pos_to_aa[pos]} vs {wt}"
            )
    
    if not pos_to_aa:
        raise ValueError("No valid mutations found to reconstruct WT")
    
    max_pos = max(pos_to_aa.keys())
    seq = []
    missing = []
    
    for i in range(1, max_pos + 1):
        if i not in pos_to_aa:
            missing.append(i)
            seq.append('X')  # Use X for missing positions
        else:
            seq.append(pos_to_aa[i])
    
    if missing:
        print(f"Warning: Missing positions filled with 'X': {missing[:10]}{'...' if len(missing) > 10 else ''}")
    
    return "".join(seq)