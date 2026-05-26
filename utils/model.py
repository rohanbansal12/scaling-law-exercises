"""Flax GPT components for the addition-transformer experiments."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from utils.data import BLOCK_SIZE, VOCAB


@dataclass(frozen=True)
class GPTConfig:
  vocab_size: int = len(VOCAB)
  block_size: int = BLOCK_SIZE
  n_layer: int = 6
  n_head: int = 6
  n_embd: int = 384
  dropout: float = 0.0
  use_bias: bool = True
  mlp_ratio: int = 4
  dtype: jnp.dtype = jnp.float32
  use_moe: bool = False
  num_experts: int = 4
  top_k: int = 1

  @property
  def head_dim(self):
    assert self.n_embd % self.n_head == 0
    return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
  config: GPTConfig

  @nn.compact
  def __call__(self, x, *, train: bool):
    B, T, C = x.shape
    assert C == self.config.n_embd

    qkv = nn.Dense(3 * C, use_bias=False)(x)

    qkv = qkv.reshape(B, T, 3, self.config.n_head, self.config.head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
    scores = q @ jnp.swapaxes(k, -1, -2)
    scores = scores / jnp.sqrt(self.config.head_dim)
    scores = jnp.where(mask, scores, -1e10)

    attn = nn.softmax(scores, axis=-1)
    attn = nn.Dropout(rate=self.config.dropout)(attn, deterministic=not train)

    out = attn @ v
    out = jnp.transpose(out, (0, 2, 1, 3))
    out = out.reshape(B, T, C)
    out = nn.Dense(C, use_bias=False)(out)
    return out


class MLP(nn.Module):
  config: GPTConfig

  @nn.compact
  def __call__(self, x, *, train: bool):
    hidden_dim = self.config.mlp_ratio * self.config.n_embd

    x = nn.Dense(hidden_dim, use_bias=False)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
    x = nn.Dense(self.config.n_embd, use_bias=False)(x)
    x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
    return x

class MoEMLP(nn.Module):
  config: GPTConfig

  @nn.compact
  def __call__(self, x, *, train: bool):
    if self.config.top_k != 1:
      raise ValueError("MoEMLP currently supports top_k=1 only.")

    B, T, C = x.shape
    E = self.config.num_experts
    H = self.config.mlp_ratio * C

    x_flat = x.reshape(B * T, C) # [M, C]

    router_logits = nn.Dense(E, use_bias=False, name="router")(x_flat) # [M, E]
    router_probs = nn.softmax(router_logits, axis=-1) # [M, E]

    expert_ids = jnp.argmax(router_probs, axis=-1) # [M]
    expert_gates = jnp.max(router_probs, axis=-1) # [M]
    sort_idx = jnp.argsort(expert_ids) # [M]
    unsort_idx = jnp.argsort(sort_idx) # [M]

    x_sorted = x_flat[sort_idx] # [M, C]
    gates_sorted = expert_gates[sort_idx, None] # [M, 1]

    group_sizes = jnp.bincount(expert_ids, length=E).astype(jnp.int32)

    W_up = self.param(
      "expert_w_up",
      nn.initializers.lecun_normal(),
      (E, C, H),
    )
    W_down = self.param(
      "expert_w_down",
      nn.initializers.lecun_normal(),
      (E, H, C),
    )

    h = jax.lax.ragged_dot(x_sorted, W_up, group_sizes)
    h = nn.gelu(h)
    h = jax.lax.ragged_dot(h, W_down, group_sizes)

    h = h * gates_sorted
    h = h[unsort_idx]
    return h.reshape(B, T, C)


class TransformerBlock(nn.Module):
  config: GPTConfig

  @nn.compact
  def __call__(self, x, *, train: bool):
    # GPT-style block: pre-norm attention, residual, pre-norm MLP, residual.
    x = x + CausalSelfAttention(config=self.config)(nn.LayerNorm()(x), train=train)
    
    if self.config.use_moe:
      ff = MoEMLP(config=self.config)(nn.LayerNorm()(x), train=train)
    else:
      ff = MLP(config=self.config)(nn.LayerNorm()(x), train=train)

    return x + ff


class GPT(nn.Module):
  config: GPTConfig

  @nn.compact
  def __call__(self, idx, *, train: bool):
    _, T = idx.shape
    assert T <= self.config.block_size

    embs = nn.Embed(self.config.vocab_size, self.config.n_embd)(idx)
    pos_emb = nn.Embed(self.config.block_size, self.config.n_embd)(jnp.arange(T))

    h = embs + pos_emb
    h = nn.Dropout(rate=self.config.dropout)(h, deterministic=not train)

    for _ in range(self.config.n_layer):
      h = TransformerBlock(config=self.config)(h, train=train)

    h = nn.LayerNorm()(h)
    return nn.Dense(self.config.vocab_size)(h)
