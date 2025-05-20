# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Loss, gradient, and update functions for predictor training/tuning."""

from collections.abc import Callable
import functools
from typing import Any

from flax import traverse_util
import jax
import jax.numpy as jnp
import optax

from thunnini.src import config as config_lib
from thunnini.src import predictor as predictor_lib
from thunnini.src import types


def log_loss(
    sequences: types.Sequences,
    log_predictions: types.LogPredictions,
) -> types.Losses:
  """Returns the cumulative log-loss for sequences."""
  cross_entropies = optax.safe_softmax_cross_entropy(log_predictions, sequences)
  return jnp.cumsum(cross_entropies, axis=1)


def apply_updates(
    params: optax.Params,
    updates: optax.Updates,
    tuning_method: types.TuningMethodType,
    prefix: types.Prefix = None,
) -> tuple[optax.Params, types.Prefix]:
  """Applies updates to parameters or prefix according to tuning method."""

  def partial_update(
      prms: optax.Params,
      updts: optax.Updates,
      update_keys: list[str],
  ):
    """Applies updates to the parameters for the given keys."""

    def should_update(path):
      """Returns True if the variable at path should be updated."""
      # Path is a tuple formed by the keys of the nested pytree, e.g.,
      # ('params', 'embedding', 'bias').
      return any(
          any(p.startswith(update_key) for p in path)
          for update_key in update_keys
      )

    updts = traverse_util.flatten_dict(updts)  # To be able to index by path.
    return traverse_util.path_aware_map(
        lambda path, x: x + updts[path] if should_update(path) else x, prms
    )

  def full_update_without_lora(
      prms: optax.Params,
      updts: optax.Updates,
      lora_param_prefix: str = 'LoRA',
  ):
    """Updates all parameters except the lora ones."""

    def should_not_update(path):
      """Returns True if the variable at path is a LoRA parameter."""
      return any(p.startswith(lora_param_prefix) for p in path)

    updts = traverse_util.flatten_dict(updts)  # To be able to index by path.
    return traverse_util.path_aware_map(
        lambda path, x: x + updts[path] if not should_not_update(path) else x,
        prms,
    )

  match tuning_method:
    case 'prefix_real' | 'prefix_simplex' | 'prefix_soft':
      # In these cases the gradient is w.r.t. the prefix, so we can apply
      # the full update to the prefix.
      return params, optax.apply_updates(prefix, updates)
    case 'full_parameters':
      return full_update_without_lora(params, updates), prefix
    case 'lora_finetune':
      return partial_update(params, updates, ['LoRA']), prefix
    case 'embedding':
      return partial_update(params, updates, ['embedding']), prefix
    case 'unembedding':
      return partial_update(params, updates, ['unembedding']), prefix
    case 'embedding_unembedding':
      return (
          partial_update(params, updates, ['embedding', 'unembedding']),
          prefix,
      )


def make_loss_fn(
    predictor: predictor_lib.Predictor,
) -> Callable[
    [optax.Params, types.Sequences, types.PrefixType, types.Prefix], jnp.float32
]:
  """Returns function that does forward-pass and computes loss."""

  def seqs_loss_fn(
      params: optax.Params,
      seqs: types.Sequences,
      prefix_type: types.PrefixType = 'none',
      prefix: types.Prefix = None,
  ) -> jnp.float32:
    """Apply and loss over sequences only, excluding the prefix."""
    logits, _, _, _ = predictor.apply(params, seqs, prefix_type, prefix)
    cross_entropies = optax.safe_softmax_cross_entropy(logits, seqs)
    return jnp.mean(jnp.sum(cross_entropies, axis=1))

  return seqs_loss_fn


def make_grad_fn(
    predictor: predictor_lib.Predictor,
    tuning_method: types.TuningMethodType = 'full_parameters',
) -> Callable[
    [optax.Params, types.Sequences, types.PrefixType, types.Prefix],
    tuple[jnp.float32, Any],
]:
  """Returns function that does forward-pass and computes gradient."""
  match tuning_method:
    case 'prefix_real' | 'prefix_simplex' | 'prefix_soft':
      return jax.value_and_grad(make_loss_fn(predictor), argnums=3)
    case 'full_parameters':
      if not predictor.torso_config.is_trainable:
        raise ValueError('Full parameter tuning requires a trainable torso.')
      return jax.value_and_grad(make_loss_fn(predictor))
    case 'lora_finetune':
      if not isinstance(
          predictor.torso_config, config_lib.TransformerTorsoConfig
      ):
        raise ValueError('LoRA tuning only supported for Transformer torsos.')
      if not predictor.torso_config.use_lora:
        raise ValueError('LoRA disabled in torso config.')
      return jax.value_and_grad(make_loss_fn(predictor))
    case 'embedding' | 'unembedding' | 'embedding_unembedding':
      return jax.value_and_grad(make_loss_fn(predictor))


@functools.partial(
    jax.jit,
    static_argnames=('grad_fn', 'optimizer', 'prefix_type', 'tuning_method'),
    donate_argnames=('params', 'opt_state', 'prefix'),
)
def update_parameters(
    params: optax.Params,
    opt_state: optax.OptState,
    sequences: types.Sequences,
    grad_fn: Callable[
        [optax.Params, types.Sequences, types.PrefixType, types.Prefix],
        tuple[jnp.float32, Any],
    ],
    optimizer: optax.GradientTransformation,
    prefix_type: types.PrefixType = 'none',
    prefix: types.Prefix = None,
    tuning_method: types.TuningMethodType = 'full_parameters',
) -> tuple[
    optax.Params, types.Prefix, optax.OptState, jnp.float32, jnp.float32
]:
  """Forward pass, then updates the `params` given `sequences` and a `grad_fn`.

  Args:
    params: The parameters of the predictor.
    opt_state: The optimizer state.
    sequences: The input sequences.
    grad_fn: The gradient function.
    optimizer: The optimizer.
    prefix_type: The type of the prefix. Can be 'none' for no prefix.
    prefix: The prefix that is prepended to sequences.
    tuning_method: The tuning method.

  Returns:
    The updated parameters or prefix, the new optimizer state, the loss, and the
    gradient norm.
  """
  loss, grad = grad_fn(params, sequences, prefix_type, prefix)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params, new_prefix = apply_updates(params, updates, tuning_method, prefix)
  grad_norm_unclipped = optax.global_norm(grad)
  return new_params, new_prefix, new_opt_state, loss, grad_norm_unclipped
