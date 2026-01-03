import pandas as pd

def reconstruct_wt_sequence(
    df: pd.DataFrame,
    mutation_col: str = "mutant",
) -> str:
    """
    Reconstruct WT sequence from single-substitution mutations like A23V.

    Assumes:
    - mutation format: WT + position + MUT (e.g. A23V)
    - positions are 1-indexed
    """
    pos_to_aa = {}

    for m in df[mutation_col].dropna().astype(str):
        if not m or len(m) < 3:
            continue

        wt = m[0]
        mut = m[-1]
        pos_str = m[1:-1]

        if not pos_str.isdigit():
            continue

        pos = int(pos_str)

        # Store WT AA for this position
        if pos not in pos_to_aa:
            pos_to_aa[pos] = wt
        else:
            # Sanity check consistency
            if pos_to_aa[pos] != wt:
                raise ValueError(
                    f"Inconsistent WT AA at position {pos}: "
                    f"{pos_to_aa[pos]} vs {wt}"
                )

    if not pos_to_aa:
        raise ValueError("Could not reconstruct WT sequence from mutations.")

    # Build sequence in order
    max_pos = max(pos_to_aa.keys())
    seq = []
    for i in range(1, max_pos + 1):
        if i not in pos_to_aa:
            raise ValueError(f"Missing WT amino acid at position {i}")
        seq.append(pos_to_aa[i])

    return "".join(seq)
