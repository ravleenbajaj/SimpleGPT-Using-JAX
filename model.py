"""
Minimal GPT implemented from scratch in JAX / Flax.

Architecture:
  - Token + learned positional embeddings
  - N stacked Transformer blocks, each with:
      * Multi-head causal self-attention (with masking)
      * Position-wise feed-forward MLP
      * Pre-norm LayerNorm (as in GPT-2)
  - Final LayerNorm + linear projection to vocabulary logits

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           GPT-2 (Radford et al., 2019)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int        # number of unique tokens
    block_size: int        # maximum context length (T)
    n_embd: int   = 128   # embedding dimension
    n_head: int   = 4     # number of attention heads (must divide n_embd)
    n_layer: int  = 4     # number of Transformer blocks
    dropout: float = 0.1  # dropout probability (applied during training)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask.

    Each token can only attend to itself and tokens before it in the sequence.
    This is enforced by masking out future positions before the softmax.
    """
    config: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, T, C = x.shape                       # batch, sequence length, embedding dim
        cfg = self.config
        head_dim = C // cfg.n_head              # dimension per head

        # Project input to queries, keys, values in one shot then split
        qkv = nn.Dense(3 * C, use_bias=False)(x)           # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)               # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim) for batched attention
        def split_heads(t):
            return t.reshape(B, T, cfg.n_head, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention scores
        scale = head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale  # (B, n_head, T, T)

        # Causal mask: upper triangle (future positions) → -inf before softmax
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn = jnp.where(causal_mask, attn, jnp.finfo(attn.dtype).min)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(cfg.dropout)(attn, deterministic=deterministic)

        # Weighted sum of values, reassemble heads
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)        # (B, n_head, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)    # (B, T, C)

        # Output projection
        out = nn.Dense(C)(out)
        out = nn.Dropout(cfg.dropout)(out, deterministic=deterministic)
        return out


class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    Expands embedding dim by 4x, applies GELU, projects back.
    This is where most of the model's capacity lives.
    """
    config: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        C = self.config.n_embd
        x = nn.Dense(4 * C)(x)
        x = nn.gelu(x)
        x = nn.Dense(C)(x)
        x = nn.Dropout(self.config.dropout)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer block: pre-norm attention + pre-norm MLP.
    Pre-norm (LayerNorm before the sublayer) stabilises training vs post-norm.
    Residual connections let gradients flow directly to earlier layers.
    """
    config: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = x + CausalSelfAttention(self.config)(nn.LayerNorm()(x), deterministic)
        x = x + MLP(self.config)(nn.LayerNorm()(x), deterministic)
        return x


class GPT(nn.Module):
    """
    Full GPT language model.

    Forward pass returns logits of shape (B, T, vocab_size).
    To compute cross-entropy loss, shift logits and targets by one position:
      - logits[:, :-1, :] predicts targets[:, 1:]
    """
    config: GPTConfig

    @nn.compact
    def __call__(
        self,
        idx: jnp.ndarray,          # token indices, shape (B, T)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        B, T = idx.shape
        assert T <= cfg.block_size, f"Sequence length {T} > block_size {cfg.block_size}"

        # Token embeddings + learned positional embeddings
        tok_emb = nn.Embed(cfg.vocab_size, cfg.n_embd)(idx)              # (B, T, C)
        pos     = jnp.arange(T)[None, :]                                  # (1, T)
        pos_emb = nn.Embed(cfg.block_size, cfg.n_embd)(pos)               # (1, T, C)

        x = nn.Dropout(cfg.dropout)(tok_emb + pos_emb, deterministic=deterministic)

        # Stack of Transformer blocks
        for _ in range(cfg.n_layer):
            x = TransformerBlock(cfg)(x, deterministic)

        # Final LayerNorm + project to vocab
        x = nn.LayerNorm()(x)
        logits = nn.Dense(cfg.vocab_size, use_bias=False)(x)   # (B, T, vocab_size)
        return logits
