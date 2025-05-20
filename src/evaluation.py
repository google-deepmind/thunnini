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

"""Evaluation functions for trained / fine tuned predictors."""

import functools

import chex
import jax
import jax.numpy as jnp
import optax

from thunnini.src import builders
from thunnini.src import config as config_lib
from thunnini.src import predictor as predictor_lib
from thunnini.src import types


def evaluate_predictor_from_datagen(
    predictor_config: config_lib.PredictorConfig,
    torso_config: config_lib.PredictorTorsoConfig,
    predictor_params: optax.Params,
    datagen_config: config_lib.DataGeneratorConfig,
    datagen_seed: int = 1337,
    datagen_num_batches: int = 1,
    prefix_type: types.PrefixType = 'none',
    prefix: types.Prefix = None,
    return_gt_and_optimal_results: bool = False,
) -> (
    tuple[types.Sequences, types.LogPredictions, types.Losses]
    | tuple[
        types.Sequences,
        types.LogPredictions,
        types.Losses,
        types.LogPredictions,
        types.Losses,
        types.LogPredictions,
        types.Losses,
    ]
):
  """Evaluates predictor and returns data, log predictions and losses.

  Draws a number of batches from the data generator, passes them through the
  predictor and evaluates the log loss. Since evaluation is generally done on
  larger amounts of data, but large batch sizes may make evaluation slow, the
  number of batches can be more than one and all results will simply be
  concatenated, as if a larger single batch had been evaluated.

  Args:
    predictor_config: Configuration for predictor.
    torso_config: Configuration for predictor torso.
    predictor_params: Parameters of predictor.
    datagen_config: Configuration for data generator.
    datagen_seed: Seed for data generation.
    datagen_num_batches: Number of batches to generate.
    prefix_type: Type of prefix to use.
    prefix: Prefix to use.
    return_gt_and_optimal_results: Whether to return ground truth and optimal
      log predictions and losses as well.

  Returns:
    sequences: Data generated for evaluation.
    logits: Log predictions for sequences.
    losses: Log losses for sequences.
    [Optiona] optimal_logits: Byaes-optimal log predictions for sequences.
    [Optiona] optimal_losses: Byaes-optimal log losses for sequences.
    [Optiona] ground_truth_logits: Ground truth log predictions.
    [Optiona] ground_truth_losses: Ground truth log losses.
  """
  predictor = builders.build_predictor(predictor_config, torso_config)
  data_generator = builders.build_datagen(datagen_config)

  def generate_and_eval_one_batch(rng_key: chex.PRNGKey):
    """Helper function to process one batch."""
    if return_gt_and_optimal_results:
      sequences, bo_log_probs, bo_losses, gt_log_probs, gt_losses = (
          data_generator.generate_solve_and_losses(rng_key=rng_key)
      )
      predictor_logits, _, _, _ = predictor.apply(
          predictor_params, sequences, prefix_type, prefix
      )
      predictor_log_losses = data_generator.instant_log_loss_from_logits(
          predictor_logits, sequences
      )
      return (
          sequences,
          predictor_logits,
          predictor_log_losses,
          bo_log_probs,
          bo_losses,
          gt_log_probs,
          gt_losses,
      )
    else:
      sequences = data_generator.generate(
          rng_key=rng_key, return_ground_truth_log_probs=False
      )
      predictor_logits, _, _, _ = predictor.apply(
          predictor_params, sequences, prefix_type, prefix
      )
      predictor_log_losses = data_generator.instant_log_loss_from_logits(
          predictor_logits, sequences
      )
      return sequences, predictor_logits, predictor_log_losses

  rng_now, rng_next = jax.random.split(jax.random.PRNGKey(datagen_seed))
  results = generate_and_eval_one_batch(rng_now)

  if datagen_num_batches > 1:
    for _ in range(datagen_num_batches - 1):
      rng_now, rng_next = jax.random.split(rng_next)
      results = jax.tree.map(
          lambda x, y: jnp.concatenate([x, y], axis=0),
          results,
          generate_and_eval_one_batch(rng_now),
      )
  return results


def evaluate_predictor_from_sequences(
    predictor_config: config_lib.PredictorConfig | None,
    torso_config: config_lib.PredictorTorsoConfig | None,
    predictor_params: optax.Params,
    sequences: types.Sequences,
    predictor_instance: predictor_lib.Predictor | None = None,
    batch_size: int = -1,
    prefix_type: types.PrefixType = 'none',
    prefix: types.Prefix = None,
) -> tuple[types.LogPredictions, types.Losses]:
  """Evaluates predictor on sequences and returns log predictions and losses.

  Since a large number of sequences may lead to a very large batch size, and
  thus slow / memory-intensive evaluation, the sequences can be split into
  multiple batches via `batch_size`. If `batch_size` is less than 1, then the
  sequences are not batched. When batching, results are concatenated so they
  appear as non-batched.

  Args:
    predictor_config: Configuration for predictor. If not provided,
      `predictor_instance` must be provided.
    torso_config: Configuration for predictor torso. If not provided,
      `predictor_instance` must be provided.
    predictor_params: Parameters of predictor.
    sequences: Sequences to evaluate on.
    predictor_instance: Predictor instance, optional (use when running many
      evaluations on the same predictor). If not provided, a new predictor is
      built from `predictor_config` and `torso_config`.
    batch_size: Number of sequences to evaluate at a time. If set to less than
      1, then the sequences are not batched. If final batch is smaller than
      `batch_size`, then final evaluation is done on remaining sequences.
    prefix_type: Type of prefix to use.
    prefix: Prefix to use.

  Returns:
    logits: Log predictions for sequences.
    losses: Log losses for sequences.
  """

  if predictor_config is not None or torso_config is not None:
    if predictor_instance is not None:
      raise ValueError(
          'Cannot provide predictor configs (`predictor_config`,'
          ' `torso_config`) and `predictor_instance` simulatenously.'
      )
    predictor = builders.build_predictor(predictor_config, torso_config)
  else:
    if predictor_instance is None:
      raise ValueError(
          'Either configs for the predictor or a `predictor_instance` must be'
          ' provided.'
      )
    predictor = predictor_instance

  def eval_one_batch(sequences_batch: types.Sequences):
    """Helper function to process one batch."""
    predictor_logits, _, _, _ = predictor.apply(
        predictor_params, sequences_batch, prefix_type, prefix
    )
    predictor_log_losses = optax.safe_softmax_cross_entropy(
        predictor_logits, sequences_batch
    )
    return predictor_logits, predictor_log_losses

  lower = 0
  upper = batch_size if batch_size > 0 else sequences.shape[0]
  results = eval_one_batch(sequences[lower:upper])

  while upper < sequences.shape[0]:
    lower += batch_size
    upper += batch_size
    if upper > sequences.shape[0]:
      upper = sequences.shape[0]

    results = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=0),
        results,
        eval_one_batch(sequences[lower:upper]),
    )
  return results


def evaluate_prefix_list(
    prefix_list: list[types.Prefix],
    prefix_type: types.PrefixType,
    predictor_config: config_lib.PredictorConfig,
    torso_config: config_lib.PredictorTorsoConfig,
    predictor_params: optax.Params,
    sequences: types.Sequences,
    batch_size: int = -1,
) -> tuple[list[types.LogPredictions], list[types.Losses]]:
  """Evaluate many different prefixes on the same predictor and sequences.

  Args:
    prefix_list: List of prefixes to evaluate.
    prefix_type: Type of prefix. Cannot be 'none'.
    predictor_config: Configuration for predictor.
    torso_config: Configuration for predictor torso.
    predictor_params: Parameters of predictor.
    sequences: Sequences to evaluate on.
    batch_size: Number of sequences to evaluate at a time. If set to less than
      1, then the sequences are not batched. If final batch is smaller than
      `batch_size`, then final evaluation is done on remaining sequences.

  Returns:
    logits_list: List of log predictions, one entry per prefix.
    losses_list: List of log losses, one entry per prefix.
  """
  predictor = builders.build_predictor(predictor_config, torso_config)
  eval_fn = functools.partial(
      evaluate_predictor_from_sequences,
      predictor_config=None,
      torso_config=None,
      predictor_params=predictor_params,
      sequences=sequences,
      predictor_instance=predictor,
      batch_size=batch_size,
      prefix_type=prefix_type,
  )

  results = [eval_fn(prefix=prefix) for prefix in prefix_list]
  logits_list, losses_list = zip(*results)
  return logits_list, losses_list


def predictions_and_states_from_sequences(
    predictor_config: config_lib.PredictorConfig | None,
    torso_config: config_lib.PredictorTorsoConfig | None,
    predictor_params: optax.Params,
    sequences: types.Sequences,
    predictor_instance: predictor_lib.Predictor | None = None,
    prefix_type: types.PrefixType = 'none',
    prefix: types.Prefix = None,
) -> tuple[
    types.LogPredictions,
    types.Hidden,
    types.PrefixLogPredictions,
    types.PrefixHidden,
]:
  """Evaluates on sequences and returns log predictions and internal states.

  Note that torso config must also be set to return internal states.

  Args:
    predictor_config: Configuration for predictor. If not provided,
      `predictor_instance` must be provided.
    torso_config: Configuration for predictor torso. If not provided,
      `predictor_instance` must be provided.
    predictor_params: Parameters of predictor.
    sequences: Sequences to evaluate on.
    predictor_instance: Predictor instance, optional (use when running many
      evaluations on the same predictor). If not provided, a new predictor is
      built from `predictor_config` and `torso_config`.
    prefix_type: Type of prefix to use.
    prefix: Prefix to use.

  Returns:
    logits: Log predictions for sequences.
    states: Internal states of torso for sequences.
    prefix_logits: Log predictions for prefix.
    prefix_states: Internal states of the torso for prefix.
  """

  if predictor_config is not None or torso_config is not None:
    if predictor_instance is not None:
      raise ValueError(
          'Cannot provide predictor configs (`predictor_config`,'
          ' `torso_config`) and `predictor_instance` simulatenously.'
      )
    predictor = builders.build_predictor(predictor_config, torso_config)
  else:
    if predictor_instance is None:
      raise ValueError(
          'Either configs for the predictor or a `predictor_instance` must be'
          ' provided.'
      )
    predictor = predictor_instance

  predictor_logits, states, prefix_logits, prefix_states = predictor.apply(
      predictor_params, sequences, prefix_type, prefix
  )

  return predictor_logits, states, prefix_logits, prefix_states
