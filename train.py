"""
Training loop for the GPT model.

Key JAX patterns used:
  - jax.jit      : compile the update step to XLA for fast execution
  - jax.value_and_grad : compute loss and gradients in one pass
  - optax        : gradient-based optimiser (AdamW with cosine LR decay)
  - flax TrainState : bundles model params + optimiser state cleanly
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Any
import time

from model import GPT, GPTConfig
from data import CharTokenizer, TextDataset, train_val_split, load_text


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Standard cross-entropy over the vocabulary.
    logits : (B, T, vocab_size)
    targets: (B, T)
    We flatten B and T together — each position is an independent prediction.
    """
    B, T, V = logits.shape
    logits  = logits[:, :-1, :].reshape(-1, V)   # predict positions 0..T-2
    targets = targets[:, 1:].reshape(-1)          # targets are positions 1..T-1
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean()


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

def create_train_state(
    rng: jax.random.PRNGKey,
    config: GPTConfig,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
) -> train_state.TrainState:
    """Initialise model parameters and optimiser."""
    model = GPT(config)

    # Dummy input to trigger parameter initialisation
    dummy = jnp.zeros((1, config.block_size), dtype=jnp.int32)
    params = model.init(rng, dummy)

    # Cosine decay with linear warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.1,
    )

    # AdamW: Adam with decoupled weight decay (better regularisation than L2)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),          # gradient clipping
        optax.adamw(schedule, weight_decay=weight_decay),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ---------------------------------------------------------------------------
# JIT-compiled train / eval steps
# ---------------------------------------------------------------------------

@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    dropout_rng: jax.random.PRNGKey,
) -> tuple:
    """Single gradient update step. Returns (new_state, loss)."""

    def loss_fn(params):
        logits = state.apply_fn(
            params, x, deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        return cross_entropy_loss(logits, y)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate loss without dropout or gradient computation."""
    logits = state.apply_fn(state.params, x, deterministic=True)
    return cross_entropy_loss(logits, y)


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(
    state: train_state.TrainState,
    config: GPTConfig,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> str:
    """
    Autoregressive generation.
    At each step:
      1. Run the model on the current context (up to block_size tokens).
      2. Take logits at the last position, scale by temperature.
      3. Sample the next token from the resulting distribution.
      4. Append and repeat.

    temperature < 1 → sharper / more confident
    temperature > 1 → flatter / more random
    temperature = 0 → greedy (argmax)
    """
    idx = jnp.array(tokenizer.encode(prompt))[None, :]   # (1, T)

    for _ in range(max_new_tokens):
        # Crop context to block_size
        idx_cond = idx[:, -config.block_size:]

        logits = state.apply_fn(state.params, idx_cond, deterministic=True)
        logits = logits[:, -1, :]    # logits for the next token only

        if temperature == 0:
            next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        else:
            logits = logits / temperature
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, logits)[:, None]

        idx = jnp.concatenate([idx, next_token], axis=1)

    return tokenizer.decode(idx[0])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    text_path: str,
    n_layer: int   = 4,
    n_head: int    = 4,
    n_embd: int    = 128,
    block_size: int = 128,
    batch_size: int = 32,
    max_iters: int  = 3000,
    eval_interval: int = 200,
    eval_iters: int    = 50,
    learning_rate: float = 3e-4,
    weight_decay: float  = 0.1,
    warmup_steps: int    = 100,
    dropout: float = 0.1,
    seed: int = 42,
):
    rng = jax.random.PRNGKey(seed)

    # --- Data ---
    text      = load_text(text_path)
    tokenizer = CharTokenizer(text)
    data      = tokenizer.encode(text)
    train_data, val_data = train_val_split(data, val_fraction=0.1)

    train_ds = TextDataset(train_data, block_size)
    val_ds   = TextDataset(val_data,   block_size)

    print(f"Corpus  : {len(text):,} characters")
    print(f"Vocab   : {tokenizer.vocab_size} unique characters")
    print(f"Train   : {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    # --- Model ---
    config = GPTConfig(
        vocab_size = tokenizer.vocab_size,
        block_size = block_size,
        n_embd     = n_embd,
        n_head     = n_head,
        n_layer    = n_layer,
        dropout    = dropout,
    )
    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(
            GPT(config).init(jax.random.PRNGKey(0), jnp.zeros((1, block_size), jnp.int32))
        )
    )
    print(f"Model   : {n_params:,} parameters\n")

    # --- Optimiser ---
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng, config, learning_rate, weight_decay, warmup_steps, max_iters
    )

    # --- Loop ---
    t0 = time.time()
    for step in range(1, max_iters + 1):
        rng, data_rng, drop_rng = jax.random.split(rng, 3)
        x, y   = train_ds.get_batch(data_rng, batch_size)
        state, loss = train_step(state, x, y, drop_rng)

        if step % eval_interval == 0 or step == 1:
            # Estimate validation loss over several batches
            val_losses = []
            rng, *val_rngs = jax.random.split(rng, eval_iters + 1)
            for vr in val_rngs:
                vx, vy = val_ds.get_batch(vr, batch_size)
                val_losses.append(float(eval_step(state, vx, vy)))
            val_loss = sum(val_losses) / len(val_losses)

            elapsed = time.time() - t0
            print(
                f"step {step:4d}/{max_iters} | "
                f"train loss {float(loss):.4f} | "
                f"val loss {val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

    print("\nTraining complete.")
    return state, tokenizer, config


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/input.txt"
    state, tokenizer, config = train(path)

    print("\n--- Sample generation (temperature=0.8) ---\n")
    sample = generate(
        state, config, tokenizer,
        prompt="\n",
        max_new_tokens=300,
        temperature=0.8,
        rng=jax.random.PRNGKey(1),
    )
    print(sample)
