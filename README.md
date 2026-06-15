# nanogpt-jax

A minimal GPT language model implemented **from scratch** in [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).

Built to understand transformer internals deeply — no magic `AutoModel.from_pretrained`, just clean NumPy-style array operations compiled to XLA.

---

## What's implemented

| Component | File | Notes |
|---|---|---|
| Multi-head causal self-attention | `model.py` | Causal mask, scaled dot-product, head splitting |
| Transformer block | `model.py` | Pre-norm, residual connections, MLP with GELU |
| Positional embeddings | `model.py` | Learned, added to token embeddings |
| Character-level tokeniser | `data.py` | No external dependency |
| JIT-compiled train step | `train.py` | `jax.jit` + `jax.value_and_grad` |
| AdamW + cosine LR schedule | `train.py` | Linear warmup via `optax` |
| Temperature sampling | `train.py` | Greedy (T=0) or stochastic |

---

## Architecture

```
Input tokens (B, T)
       │
  Token Embedding (vocab_size → n_embd)
+ Positional Embedding (block_size → n_embd)
       │
  ┌────┴────────────────────┐
  │   Transformer Block ×N  │
  │  ┌─────────────────┐    │
  │  │   LayerNorm     │    │
  │  │   CausalAttn   │    │
  │  │   + Residual    │    │
  │  ├─────────────────┤    │
  │  │   LayerNorm     │    │
  │  │      MLP        │    │
  │  │   + Residual    │    │
  │  └─────────────────┘    │
  └─────────────────────────┘
       │
  LayerNorm
       │
  Linear → logits (B, T, vocab_size)
```

---

## Quickstart

```bash
pip install jax flax optax
```

### Smoke test (no data needed)

```bash
python demo.py
```

Trains a tiny 2-layer model on a short text snippet for 100 steps and prints generated text.

### Train on your own text

```bash
# Put any plain .txt file in data/
python train.py data/input.txt
```

A good dataset to try: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB).

---

## Key JAX patterns

**`jax.jit`** — compiles the entire train step to XLA, making it run significantly faster than eager execution:
```python
@jax.jit
def train_step(state, x, y, dropout_rng):
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
```

**`jax.value_and_grad`** — computes loss and gradients in a single forward+backward pass. No `.backward()` call needed.

**Functional state** — JAX has no mutable state. Model parameters and optimiser state live in a `TrainState` pytree that is explicitly threaded through each step.

---

## Hyperparameters (defaults)

| Parameter | Default | Description |
|---|---|---|
| `n_layer` | 4 | Number of Transformer blocks |
| `n_head` | 4 | Number of attention heads |
| `n_embd` | 128 | Embedding dimension |
| `block_size` | 128 | Context window length |
| `batch_size` | 32 | Sequences per gradient step |
| `max_iters` | 3000 | Training steps |
| `learning_rate` | 3e-4 | Peak LR (cosine decay) |
| `dropout` | 0.1 | Dropout probability |

## Results

Trained on Tiny Shakespeare (~1M characters, 65-char vocab, 825K parameters):

| Step | Train loss | Val loss |
|------|-----------|----------|
| 1    | 4.96      | 4.98     |
| 1000 | 2.78      | 2.79     |
| 3000 | 2.63      | 2.68     |

Train and val loss track closely throughout, indicating no significant overfitting.
Generated sample after 3000 steps (temperature=0.8):
---

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://openai.com/research/gpt-2) (GPT-2)
- Karpathy (2022). [nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for this project
- [JAX documentation](https://jax.readthedocs.io/)
- [Flax documentation](https://flax.readthedocs.io/)
