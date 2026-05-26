"""Evaluation helpers shared by training and scaling-law sweeps."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from utils.data import pad_token


def count_params(params):
  return sum(x.size for x in jax.tree_util.tree_leaves(params))


def count_active_params(params, config):
  total_params = count_params(params)
  if not getattr(config, "use_moe", False):
    return total_params

  num_experts = getattr(config, "num_experts", 1)
  top_k = getattr(config, "top_k", 1)
  expert_total = 0
  expert_active = 0

  for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    path_text = "/".join(str(part) for part in path)
    if "expert_w_up" in path_text or "expert_w_down" in path_text:
      expert_total += leaf.size
      expert_active += leaf.size * min(top_k, num_experts) // num_experts

  return total_params - expert_total + expert_active


def cross_entropy_loss(logits, targets, loss_mask=None):
  log_probs = nn.log_softmax(logits, axis=-1)
  target_log_probs = jnp.take_along_axis(
    log_probs,
    indices=targets[..., None],
    axis=-1,
  ).squeeze(-1)
  if loss_mask is None:
    loss_mask = targets != pad_token
  loss_mask = loss_mask.astype(logits.dtype)

  denom = jnp.maximum(loss_mask.sum(), 1)
  return -(target_log_probs * loss_mask).sum() / denom


@jax.jit
def eval_step(state, batch):
  x, y, loss_mask = batch
  logits = state.apply_fn({"params": state.params}, x, train=False)
  return cross_entropy_loss(logits, y, loss_mask)


def evaluate_loss(state, x_eval, y_eval, eval_loss_masks, *, batch_size=256):
  losses = []
  for start in range(0, x_eval.shape[0], batch_size):
    batch = (
      x_eval[start:start + batch_size],
      y_eval[start:start + batch_size],
      eval_loss_masks[start:start + batch_size],
    )
    losses.append(eval_step(state, batch))
  return float(jnp.mean(jnp.array(losses)))
