"""
Data utilities: character-level tokeniser and batch sampler.

We use a character-level vocabulary for simplicity — no external tokeniser
needed. Every unique character in the training text becomes a token.
This keeps the vocab small and lets us train on any plain-text file.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Maps characters ↔ integer ids.
    Vocabulary is built from the unique characters in the training corpus.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # char → id
        self.itos = {i: ch for i, ch in enumerate(chars)}   # id  → char

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset:
    """
    Wraps a tokenised text corpus and provides random batch sampling.

    Each batch item is a pair (x, y) where:
      x = token ids at positions [t, t+block_size)
      y = token ids at positions [t+1, t+block_size+1)   (shifted by 1)

    The model is trained to predict y[t] given x[0..t].
    """

    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(
        self,
        rng: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample a random batch of (input, target) pairs."""
        ix = jax.random.randint(
            rng, shape=(batch_size,), minval=0, maxval=len(self)
        )
        x = jnp.stack([jnp.array(self.data[i : i + self.block_size])     for i in ix])
        y = jnp.stack([jnp.array(self.data[i + 1 : i + self.block_size + 1]) for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# Train / val split helper
# ---------------------------------------------------------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_val_split(
    data: np.ndarray,
    val_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    split = int(len(data) * (1 - val_fraction))
    return data[:split], data[split:]
