"""
demo.py — quick sanity check without needing a data file.

Trains on a tiny synthetic corpus (repeated Shakespeare-like text) for a
handful of steps to verify the model runs end-to-end, then generates a
short sample. Useful for CI checks and for showing the model in a notebook.
"""

import jax
import jax.numpy as jnp
from model import GPT, GPTConfig
from data import CharTokenizer, TextDataset, train_val_split
from train import create_train_state, train_step, eval_step, generate

TINY_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 40   # repeat so we have enough tokens

def run_demo(steps: int = 100, temperature: float = 0.8):
    rng = jax.random.PRNGKey(0)

    tokenizer = CharTokenizer(TINY_TEXT)
    data      = tokenizer.encode(TINY_TEXT)
    train_data, val_data = train_val_split(data, val_fraction=0.1)

    block_size = 64
    config = GPTConfig(
        vocab_size  = tokenizer.vocab_size,
        block_size  = block_size,
        n_embd      = 64,
        n_head      = 2,
        n_layer     = 2,
        dropout     = 0.0,
    )

    train_ds = TextDataset(train_data, block_size)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng, config,
        learning_rate=3e-3,
        weight_decay=0.1,
        warmup_steps=10,
        total_steps=steps,
    )

    print(f"Vocab size  : {tokenizer.vocab_size}")
    print(f"Training for {steps} steps...\n")

    for step in range(1, steps + 1):
        rng, data_rng, drop_rng = jax.random.split(rng, 3)
        x, y = train_ds.get_batch(data_rng, batch_size=16)
        state, loss = train_step(state, x, y, drop_rng)
        if step % 20 == 0:
            print(f"  step {step:3d} | loss {float(loss):.4f}")

    print("\n--- Generated text ---\n")
    out = generate(
        state, config, tokenizer,
        prompt="To be",
        max_new_tokens=150,
        temperature=temperature,
        rng=jax.random.PRNGKey(99),
    )
    print(out)


if __name__ == "__main__":
    run_demo()
