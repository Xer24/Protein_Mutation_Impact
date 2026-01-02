"""
Sequence construction utilities

Implements:
1) Mutation parsing (e.g., "A23V")
2) Generate mutant sequences from WT + mutation(s)
3) Mutation scan generator (all 19 substitutions at a position, or all positions)

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

# 20 standard amino acids
AA20 = set("ACDEFGHIKLMNPQRSTVWY")

# Match a single substitution like A23V (from_aa)(position)(to_aa)
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class Mutation:
    """A single substitution mutation."""
    from_aa: str
    pos: int  # 1-indexed
    to_aa: str

    def __str__(self) -> str:
        return f"{self.from_aa}{self.pos}{self.to_aa}"


# ---------------------------
# 1) Implement mutation parsing
# ---------------------------

def parse_mutation(mut: str) -> Mutation:
    """
    Parse a ProteinGym-style single substitution, e.g. "A23V".

    Raises:
        ValueError: if the mutation string is invalid.
    """
    if not isinstance(mut, str):
        raise ValueError(f"Mutation must be a string, got {type(mut)}")

    s = mut.strip()
    m = MUT_RE.match(s)
    if not m:
        raise ValueError(f"Invalid mutation format: {mut!r}. Expected like 'A23V'.")

    from_aa, pos_str, to_aa = m.group(1), m.group(2), m.group(3)
    pos = int(pos_str)

    # Extra sanity: don't allow from==to (not harmful, but usually meaningless)
    if from_aa == to_aa:
        raise ValueError(f"Invalid mutation (from==to): {mut!r}")

    return Mutation(from_aa=from_aa, pos=pos, to_aa=to_aa)


def parse_mutations(muts: Union[str, Sequence[str]]) -> List[Mutation]:
    """
    Parse one or many mutations.

    Accepts:
      - "A23V" (single string) -> [Mutation(...)]
      - ["A23V", "G150D"] -> [Mutation(...), Mutation(...)]
    """
    if isinstance(muts, str):
        return [parse_mutation(muts)]
    return [parse_mutation(m) for m in muts]


# ---------------------------
# Helpers: sequence validation
# ---------------------------

def validate_wt_sequence(wt_seq: str) -> None:
    """
    Validate WT sequence contains only the 20 standard amino acids.
    """
    if not isinstance(wt_seq, str) or len(wt_seq) == 0:
        raise ValueError("WT sequence must be a non-empty string.")
    bad = {c for c in wt_seq if c not in AA20}
    if bad:
        raise ValueError(f"WT sequence contains illegal amino acids: {sorted(bad)}")


# ---------------------------
# 2) Generate mutant sequences
# ---------------------------

def apply_mutation(wt_seq: str, mut: Union[str, Mutation], strict: bool = True) -> str:
    """
    Apply a single substitution to a WT sequence.

    Args:
        wt_seq: wild-type sequence
        mut: "A23V" or Mutation(from_aa="A", pos=23, to_aa="V")
        strict: if True, require that WT[pos-1] == from_aa. Strongly recommended.

    Returns:
        mutant sequence string

    Raises:
        ValueError: if position out of range, illegal AA, or mismatch (strict=True).
    """
    validate_wt_sequence(wt_seq)
    m = parse_mutation(mut) if isinstance(mut, str) else mut

    if not (1 <= m.pos <= len(wt_seq)):
        raise ValueError(f"Mutation {m} out of range for WT length {len(wt_seq)}.")

    i = m.pos - 1  # 1-indexed -> 0-indexed
    if strict and wt_seq[i] != m.from_aa:
        raise ValueError(
            f"WT mismatch for {m}: expected WT[{m.pos}]='{m.from_aa}', saw '{wt_seq[i]}'."
        )

    # Even if strict=False, we still ensure mutation AAs are valid
    if (m.from_aa not in AA20) or (m.to_aa not in AA20):
        raise ValueError(f"Mutation {m} contains illegal amino acid(s).")

    return wt_seq[:i] + m.to_aa + wt_seq[i + 1:]


def apply_mutations(
    wt_seq: str,
    muts: Union[str, Sequence[str], Sequence[Mutation]],
    strict: bool = True,
    allow_same_position: bool = False,
) -> str:
    """
    Apply multiple substitutions to a WT sequence.

    Notes:
    - By default, disallows two mutations at the same position (common bug source).
    - Applies in ascending position order for determinism.

    Args:
        wt_seq: wild-type sequence
        muts: "A23V" or ["A23V","G150D"] or [Mutation(...), ...]
        strict: require WT[pos-1] == from_aa, checked against the *current* sequence as you apply
        allow_same_position: if False, raises error if same pos appears twice

    Returns:
        mutated sequence
    """
    validate_wt_sequence(wt_seq)

    # Normalize to list[Mutation]
    if isinstance(muts, str):
        parsed: List[Mutation] = [parse_mutation(muts)]
    else:
        parsed = [parse_mutation(m) if isinstance(m, str) else m for m in muts]

    # Check duplicates
    if not allow_same_position:
        positions = [m.pos for m in parsed]
        if len(set(positions)) != len(positions):
            raise ValueError(f"Duplicate mutation positions found: {positions}")

    # Apply sorted
    seq = wt_seq
    for m in sorted(parsed, key=lambda x: x.pos):
        seq = apply_mutation(seq, m, strict=strict)
    return seq


# ---------------------------
# 3) Implement mutation scan generator
# ---------------------------

def scan_position(
    wt_seq: str,
    pos: int,
) -> List[str]:
    """
    Generate all 19 single-substitution mutations at a given position.

    Returns:
        list of mutation strings like ["A23C", "A23D", ...] excluding WT AA -> WT AA
    """
    validate_wt_sequence(wt_seq)
    if not (1 <= pos <= len(wt_seq)):
        raise ValueError(f"Position {pos} out of range for WT length {len(wt_seq)}.")

    wt_aa = wt_seq[pos - 1]
    return [f"{wt_aa}{pos}{aa}" for aa in sorted(AA20) if aa != wt_aa]


def scan_positions(
    wt_seq: str,
    positions: Iterable[int],
) -> List[str]:
    """
    Generate all 19 substitutions for a set of positions.
    """
    muts: List[str] = []
    for p in positions:
        muts.extend(scan_position(wt_seq, int(p)))
    return muts


def scan_all_positions(wt_seq: str) -> List[str]:
    """
    Generate all single substitutions across all positions (L * 19).
    """
    validate_wt_sequence(wt_seq)
    muts: List[str] = []
    for pos in range(1, len(wt_seq) + 1):
        muts.extend(scan_position(wt_seq, pos))
    return muts


def mutations_to_sequences(
    wt_seq: str,
    mutation_strings: Sequence[str],
    strict: bool = True,
) -> List[str]:
    """
    Convenience: convert a list of mutation strings into the corresponding mutant sequences.
    """
    return [apply_mutation(wt_seq, m, strict=strict) for m in mutation_strings]


