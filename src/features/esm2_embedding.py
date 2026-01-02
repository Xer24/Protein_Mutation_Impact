from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class ESM2Config:
    # Start with the small one you mentioned
    model_name: str = "esm2_t6_8M_UR50D"

    # Embedding pooling choice
    pooling: str = "mean"  # only "mean" implemented here

    # Cache path (joblib dict: {sequence_str: np.ndarray})
    cache_path: Path = Path("artifacts/esm2_cache.joblib")

    # Device preferences
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Embedding batching
    batch_size: int = 32

    # Safety: ESM2 supports long sequences, but memory can blow up.
    # If you want to enforce a max length, set this (None = no truncation).
    max_len: Optional[int] = None


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon Metal
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_esm2(model_name: str, device: torch.device):
    """
    Loads an ESM2 model from fair-esm.
    Returns (model, alphabet, batch_converter, repr_layer_idx).
    """
    import esm  # fair-esm

    if not hasattr(esm.pretrained, model_name):
        raise ValueError(
            f"Unknown ESM2 model_name='{model_name}'. "
            f"Check esm.pretrained.* names or start with 'esm2_t6_8M_UR50D'."
        )

    model_loader = getattr(esm.pretrained, model_name)
    model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()

    model.eval()
    model = model.to(device)

    # ESM2 repr layers are numbered 1..num_layers; "last layer" is num_layers
    repr_layer = model.num_layers
    return model, alphabet, batch_converter, repr_layer


def _mean_pool_tokens(token_reps: torch.Tensor, tokens: torch.Tensor, alphabet) -> torch.Tensor:
    """
    Mean-pool over residue positions only (exclude special tokens).
    token_reps: (B, T, C)
    tokens:     (B, T)
    returns:    (B, C)
    """
    # Special token indices
    pad_idx = alphabet.padding_idx
    cls_idx = alphabet.cls_idx
    eos_idx = alphabet.eos_idx

    # mask: True where token is a real residue, not PAD/CLS/EOS
    is_residue = (tokens != pad_idx) & (tokens != cls_idx) & (tokens != eos_idx)  # (B, T)

    # avoid division by zero
    lengths = is_residue.sum(dim=1).clamp(min=1)  # (B,)

    # zero out non-residue token reps
    masked = token_reps * is_residue.unsqueeze(-1)  # (B, T, C)

    pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)  # (B, C)
    return pooled


def _load_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        obj = joblib.load(cache_path)
        if isinstance(obj, dict):
            return obj
    return {}


def _save_cache(cache: Dict[str, np.ndarray], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, cache_path)


@torch.no_grad()
def embed_sequences_esm2(
    seqs: List[str],
    cfg: Optional[ESM2Config] = None,
) -> np.ndarray:
    """
    Compute ESM2 embeddings for sequences using mean-pooling of last layer token reps.
    Uses a joblib cache keyed by the exact sequence string.

    Returns:
        embeddings: np.ndarray of shape (N, D)
    """
    if cfg is None:
        cfg = ESM2Config()

    device = _resolve_device(cfg.device)
    cache = _load_cache(cfg.cache_path)

    # Optionally truncate sequences (not recommended unless you must)
    if cfg.max_len is not None:
        seqs = [s[: cfg.max_len] for s in seqs]

    # Figure out which ones we need to compute
    need = [s for s in seqs if s not in cache]

    if len(need) > 0:
        model, alphabet, batch_converter, repr_layer = _load_esm2(cfg.model_name, device)

        # Process in batches
        for start in tqdm(range(0, len(need), cfg.batch_size), desc="ESM2 embedding"):
            batch_seqs = need[start : start + cfg.batch_size]

            # fair-esm expects list of (name, sequence)
            batch = [(f"seq_{start+i}", s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(batch)  # tokens: (B, T)
            tokens = tokens.to(device)

            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_reps = out["representations"][repr_layer]  # (B, T, C)

            if cfg.pooling != "mean":
                raise ValueError(f"Unsupported pooling='{cfg.pooling}'. Only 'mean' implemented.")

            pooled = _mean_pool_tokens(token_reps, tokens, alphabet)  # (B, C)
            pooled_np = pooled.detach().cpu().numpy().astype(np.float32)

            # write into cache
            for s, vec in zip(batch_seqs, pooled_np):
                cache[s] = vec

        _save_cache(cache, cfg.cache_path)

    # Build output array in original order
    embeddings = np.stack([cache[s] for s in seqs], axis=0).astype(np.float32)
    return embeddings
