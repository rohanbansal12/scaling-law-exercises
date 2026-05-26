"""Dataset and token helpers for fixed-width addition examples."""

import jax
import jax.numpy as jnp


VOCAB = "0123456789 +=P"
PAD = "P"
MAX_NUMBER = 999
FULL_SEQ_LEN = len("999+999=1998 ") + 1
BLOCK_SIZE = FULL_SEQ_LEN - 1

stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
pad_token = stoi[PAD]


def encode(text):
  return [stoi[ch] for ch in text]


def decode(tokens):
  return "".join(itos[int(t)] for t in tokens)


def make_addition_example(a, b, *, full_seq_len=FULL_SEQ_LEN):
  text = f"{int(a)}+{int(b)}={int(a) + int(b)} "
  assert len(text) <= full_seq_len
  return encode(text.ljust(full_seq_len, PAD))


def make_addition_dataset(num_samples, rng, *, max_number=MAX_NUMBER, full_seq_len=FULL_SEQ_LEN):
  a_key, b_key = jax.random.split(rng)
  a = jax.device_get(jax.random.randint(a_key, (num_samples,), 0, max_number + 1))
  b = jax.device_get(jax.random.randint(b_key, (num_samples,), 0, max_number + 1))
  examples = jnp.array(
    [make_addition_example(ai, bi, full_seq_len=full_seq_len) for ai, bi in zip(a, b)],
    dtype=jnp.int32,
  )
  x = examples[:, :-1]
  y = examples[:, 1:]

  # Supervise only answer digits and the real trailing space after the answer.
  eq_positions = jnp.argmax(examples == stoi["="], axis=1)
  target_positions = jnp.arange(full_seq_len - 1)[None, :]
  loss_mask = (target_positions >= eq_positions[:, None]) & (y != pad_token)
  return x, y, loss_mask


def get_batch(x_data, y_data, loss_masks, batch_size, rng):
  idx = jax.random.randint(rng, (batch_size,), 0, x_data.shape[0])
  return x_data[idx], y_data[idx], loss_masks[idx]

