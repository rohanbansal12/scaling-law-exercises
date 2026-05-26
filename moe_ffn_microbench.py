import functools
import math
import time

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp


def fused_single_expert_kernel(
  x_ref,
  w_up_ref,
  w_down_ref,
  group_size_ref,
  y_ref,
  *,
  D: int,
  F: int,
  BLOCK_M: int,
  BLOCK_F: int,
):
  block_id = pl.program_id(0)
  group_size = group_size_ref[pl.dslice(0, 1)][0]
  has_work = block_id * BLOCK_M < group_size

  def do_work(_):
    token_start = block_id * BLOCK_M
    acc = jnp.zeros((BLOCK_M, D), dtype=jnp.float32)
    x_tile = x_ref[pl.dslice(token_start, BLOCK_M), pl.dslice(0, D)]

    for f0 in range(0, F, BLOCK_F):
      h = pl.dot(x_tile, w_up_ref[pl.dslice(0, D), pl.dslice(f0, BLOCK_F)])
      h = jax.nn.gelu(h)
      acc += pl.dot(h, w_down_ref[pl.dslice(f0, BLOCK_F), pl.dslice(0, D)])

    y_ref[pl.dslice(token_start, BLOCK_M), pl.dslice(0, D)] = acc.astype(y_ref.dtype)

  jax.lax.cond(has_work, do_work, lambda _: None, operand=None)


def per_expert_pallas_ffn(
  x_sorted,
  w_up,
  w_down,
  group_sizes,
  *,
  BLOCK_M=256,
  BLOCK_F=128,
):
  E, D, F = w_up.shape
  M = x_sorted.shape[0]
  if F % BLOCK_F != 0:
    raise ValueError(f"F={F} must be divisible by BLOCK_F={BLOCK_F}.")

  blocks_per_expert = math.ceil(M / BLOCK_M)
  expert_stride = blocks_per_expert * BLOCK_M

  group_starts = jnp.concatenate([
    jnp.zeros((1,), dtype=group_sizes.dtype),
    jnp.cumsum(group_sizes[:-1]),
  ])
  expert_ids_sorted = jnp.repeat(
    jnp.arange(E, dtype=jnp.int32),
    group_sizes,
    total_repeat_length=M,
  )
  group_starts_sorted = jnp.repeat(
    group_starts,
    group_sizes,
    total_repeat_length=M,
  )
  local_offsets = jnp.arange(M, dtype=jnp.int32) - group_starts_sorted
  y_sorted = jnp.zeros((M, D), dtype=x_sorted.dtype)

  kernel = functools.partial(
    fused_single_expert_kernel,
    D=D,
    F=F,
    BLOCK_M=BLOCK_M,
    BLOCK_F=BLOCK_F,
  )

  for e in range(E):
    mask = expert_ids_sorted == e
    x_e = (
      jnp.zeros((expert_stride, D), dtype=x_sorted.dtype)
      .at[local_offsets]
      .add(jnp.where(mask[:, None], x_sorted, 0))
    )
    y_e = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((expert_stride, D), x_sorted.dtype),
      grid=(blocks_per_expert,),
    )(
      x_e,
      w_up[e],
      w_down[e],
      jnp.array([group_sizes[e]], dtype=jnp.int32),
    )
    y_sorted = y_sorted + jnp.where(mask[:, None], y_e[local_offsets], 0)

  return y_sorted


def baseline_ragged_ffn(x_sorted, w_up, w_down, group_sizes):
  h = jax.lax.ragged_dot(x_sorted, w_up, group_sizes)
  h = jax.nn.gelu(h)
  return jax.lax.ragged_dot(h, w_down, group_sizes)


def time_fn(fn, xs, w_up, w_down, group_sizes, *, warmup=3):
  for x in xs[:warmup]:
    jax.block_until_ready(fn(x, w_up, w_down, group_sizes))

  times = []
  outputs = []
  for x in xs:
    start = time.perf_counter()
    y = fn(x, w_up, w_down, group_sizes)
    y = jax.block_until_ready(y)
    times.append(time.perf_counter() - start)
    outputs.append(y)
  return times, outputs


def run_case(name, group_sizes, *, num_batches=12):
  M = 512 * 13
  E = 4
  D = 512
  F = 4096
  assert int(group_sizes.sum()) == M

  key = jax.random.key(abs(hash(name)) & 0xFFFF_FFFF)
  key, w_up_key, w_down_key = jax.random.split(key, 3)
  w_up = jax.random.normal(w_up_key, (E, D, F), dtype=jnp.float32) * (D ** -0.5)
  w_down = jax.random.normal(w_down_key, (E, F, D), dtype=jnp.float32) * (F ** -0.5)

  xs = []
  for _ in range(num_batches):
    key, x_key = jax.random.split(key)
    xs.append(jax.random.normal(x_key, (M, D), dtype=jnp.float32))

  group_sizes = jax.device_put(group_sizes.astype(jnp.int32))
  w_up = jax.device_put(w_up)
  w_down = jax.device_put(w_down)
  xs = [jax.device_put(x) for x in xs]

  baseline_jit = jax.jit(baseline_ragged_ffn)
  pallas_jit = jax.jit(per_expert_pallas_ffn)

  baseline_times, baseline_outputs = time_fn(baseline_jit, xs, w_up, w_down, group_sizes)
  pallas_times, pallas_outputs = time_fn(pallas_jit, xs, w_up, w_down, group_sizes)

  diffs = [
    float(jnp.max(jnp.abs(a - b)))
    for a, b in zip(baseline_outputs, pallas_outputs)
  ]
  baseline_mean = sum(baseline_times) / len(baseline_times)
  pallas_mean = sum(pallas_times) / len(pallas_times)

  print(f"\n{name}")
  print("group_sizes:", [int(x) for x in group_sizes])
  print(f"baseline ragged_dot+gelu+ragged_dot: {baseline_mean * 1e3:.3f} ms")
  print(f"per-expert pallas ffn:              {pallas_mean * 1e3:.3f} ms")
  print(f"speedup: {baseline_mean / pallas_mean:.3f}x")
  print(f"max_abs_diff: {max(diffs)}")


def main():
  print("backend:", jax.default_backend())
  print("devices:", jax.devices())
  run_case("balanced", jnp.array([1664, 1664, 1664, 1664], dtype=jnp.int32))
  run_case("skewed", jnp.array([512, 1024, 2048, 3072], dtype=jnp.int32))


if __name__ == "__main__":
  main()
