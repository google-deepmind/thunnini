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

"""Implements prefix tuning and predictor fine tuning."""

import datetime
import functools
import time
import typing
from typing import Any

from absl import logging
import jax
from jax import random as jrandom
import jax.numpy as jnp
import optax

from thunnini.src import builders
from thunnini.src import config as config_lib
from thunnini.src import evaluation
from thunnini.src import predictor_tuning_functions
from thunnini.src import types


def generate_prefix(
    prefix_length: int,
    prefix_dimensionality: int,
    prefix_init_method: types.PrefixInitMethod,
    rng_key: jax.Array,
) -> types.Prefix:
  """Generates a (random) prefix according to the given method."""
  match prefix_init_method:
    case 'zeros':
      return jnp.zeros((prefix_length, prefix_dimensionality))
    case 'simplex':
      return jrandom.uniform(rng_key, (prefix_length, prefix_dimensionality))
    case 'one_hot':
      rvals = jrandom.uniform(rng_key, (prefix_length, prefix_dimensionality))
      return jax.nn.one_hot(jnp.argmax(rvals, axis=-1), prefix_dimensionality)


def tune(
    tuning_config: config_lib.TuningConfig,
    predictor_config: config_lib.PredictorConfig,
    torso_config: config_lib.PredictorTorsoConfig,
    predictor_params: optax.Params,
    data_config: config_lib.DataGeneratorConfig,
    initial_prefix: types.Prefix = None,
    override_datagen_seed: int | None = None,
) -> tuple[optax.Params, optax.Params, dict[str, Any]]:
  """Tunes prefix or model and returns tuned params and tuning loss curve.

  Args:
    tuning_config: Configuration for tuning.
    predictor_config: Configuration for predictor.
    torso_config: Configuration for predictor torso.
    predictor_params: Initial predictor parameters (from pretraining or random).
    data_config: Configuration for data generator.
    initial_prefix: (Optional) Initial prefix. If not provided, prefix will be
      initialized according to tuning_config.prefix_init_method. If provided,
      `prefix_init_method` in tuning config is ignored.
    override_datagen_seed: (Optional) If provided, this seed is used for the
      data generator instead of the one in tuning_config. Useful for different
      tuning runs using different samples across repetitions.

  Returns:
    Tuple of (tuned predictor parameters, tuned prefix, tuning loss curve).
  """
  tuning_method = tuning_config.tuning_method

  if not torso_config.is_trainable and tuning_method == 'full_parameters':
    raise ValueError('Full parameter tuning requires a trainable torso.')

  if tuning_method not in typing.get_args(types.TuningMethodType):
    raise ValueError(f'Unknown tuning method: {tuning_method}')

  if override_datagen_seed is not None:
    data_rng = jrandom.PRNGKey(override_datagen_seed)
  else:
    data_rng = jrandom.PRNGKey(tuning_config.data_gen_seed)
  data_generator = builders.build_datagen(data_config)
  predictor = builders.build_predictor(predictor_config, torso_config)

  optimizer = optax.chain(
      optax.clip_by_global_norm(tuning_config.max_grad_norm),
      optax.adam(tuning_config.learning_rate),
  )

  match tuning_method:
    case 'prefix_real':
      prefix_type = 'prepend'
      prefix_dim = predictor_config.token_dimensionality
    case 'prefix_simplex':
      prefix_type = 'simplex'
      prefix_dim = predictor_config.token_dimensionality
    case 'prefix_soft':
      prefix_type = 'embedding'
      prefix_dim = predictor_config.embedding_dimensionality
    case _:
      prefix_type = 'none'
      prefix_dim = 0

  if tuning_method in ['prefix_real', 'prefix_simplex', 'prefix_soft']:
    if initial_prefix is None:
      initial_prefix = generate_prefix(
          prefix_length=tuning_config.prefix_length,
          prefix_dimensionality=prefix_dim,
          prefix_init_method=tuning_config.prefix_init_method,
          rng_key=jrandom.PRNGKey(tuning_config.prefix_init_seed),
      )

    prefix = initial_prefix
    opt_state = optimizer.init(params=prefix)
  else:
    prefix = None
    opt_state = optimizer.init(params=predictor_params)

  grad_fn = predictor_tuning_functions.make_grad_fn(
      predictor=predictor, tuning_method=tuning_config.tuning_method
  )
  update_fn = functools.partial(
      predictor_tuning_functions.update_parameters,
      grad_fn=grad_fn,
      optimizer=optimizer,
      prefix_type=prefix_type,
      tuning_method=tuning_config.tuning_method,
  )

  # Make a copy of parameters and prefix, otherwise they will get deleted since
  # they are marked as `donate` argnames in `update_parameters`.
  predictor_params_cp = jax.tree_util.tree_map(jnp.copy, predictor_params)
  prefix_cp = jax.tree_util.tree_map(jnp.copy, prefix)

  results = {
      'loss': [],
  }
  logging.info('Tuning for %s steps.', tuning_config.num_tuning_steps)
  start_time = time.process_time()
  for _ in range(0, tuning_config.num_tuning_steps + 1):

    data_rng, data_rng_now = jrandom.split(data_rng)
    sequences = data_generator.generate(
        rng_key=data_rng_now, return_ground_truth_log_probs=False
    )

    predictor_params_cp, prefix_cp, opt_state, loss, _ = update_fn(
        params=predictor_params_cp,
        opt_state=opt_state,
        sequences=sequences,
        prefix=prefix_cp,
    )

    results['loss'].append(loss)

  logging.info(
      'Tuning finished after %s seconds.',
      datetime.timedelta(seconds=time.process_time() - start_time),
  )
  results['initial_prefix'] = initial_prefix
  return predictor_params_cp, prefix_cp, results


def run_tuning_experiment(
    predictor_config: config_lib.PredictorConfig,
    torso_config: config_lib.PredictorTorsoConfig,
    predictor_params: optax.Params,
    tuning_configs: dict[str, config_lib.TuningConfig],
    tuning_data_config: config_lib.DataGeneratorConfig,
    eval_data_configs: dict[str, config_lib.DataGeneratorConfig],
    eval_datgen_seed: int = 1337,
    eval_batching_batch_size: int = -1,
    evaluate_untuned_predictor: bool = False,
    return_tuned_params: bool = False,
    return_tuned_prefix: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Tunes a predictor with various methods and evaluates on data generators.

  This function takes a given predictor and tunes it with various methods
  (e.g., prefix tuning, embedding tuning, etc.) on a single tuning data
  distribution specified by the `tuning_data_config`. The tuning can be repeated
  a number of times with different initial conditions, depending on the tuning
  config. The tuned predictor is then evaluated on potentially multiple
  evaluation data generators. All results are collected and returned in a
  dictionary, the sequences used for evaluation are returned as well.

  Args:
    predictor_config: Configuration of predictor.
    torso_config: Configuration of predictor torso.
    predictor_params: Initial predictor parameters (from pretraining or random).
    tuning_configs: Dictionary of tuning configurations, each tuning
      configuration leads to a separate tuning and evaluation, with all results
      collected in the results dictionary.
    tuning_data_config: Data generator config for tuning.
    eval_data_configs: Dictionary of evaluation data generator configurations.
    eval_datgen_seed: Seed for data generator used for evaluation.
    eval_batching_batch_size: For performance reasons, the evaluation batch
      drawn from the evaluation data generator can be split into multiple
      batches. This argument specifies the size of each batch. If set to less
      than 1, then all data is processed in a single batch.
    evaluate_untuned_predictor: Whether to evaluate the predictor without any
      tuning and add its results to the results dictionary.
    return_tuned_params: Whether to return tuned predictor parameters.
    return_tuned_prefix: Whether to return the initial and tuned prefix.

  Returns:
    results: Dictionary of results, containing tuning and evaluation results.
      The dictionary has one entry per tuning method, which itself is a dict
      with one entry per evaluation data source. The latter has a list of
      entries, one per tuning repetition.
    sequences: Dictionary containing batch of sequences used for evaluation for
      each evaluation data source.
  """

  results = {}
  results['Bayes'] = {}
  results['ground_truth'] = {}
  if evaluate_untuned_predictor:
    results['NoTuning'] = {}
  sequences = {}
  prefix_tuning_methods = ['prefix_real', 'prefix_simplex', 'prefix_soft']

  # Run through all tuning methods as specified in tuning_configs dict.
  for i, (tuning_name, tuning_config) in enumerate(tuning_configs.items()):
    logging.info(
        '---- %s tuning (method %d of %d) ----',
        tuning_config.tuning_method,
        i + 1,
        len(tuning_configs),
    )
    prefix_init_seed = jrandom.PRNGKey(tuning_config.prefix_init_seed)
    datagen_seed = tuning_config.data_gen_seed
    match tuning_config.tuning_method:
      case 'prefix_real':
        prefix_type = 'prepend'
        prefix_dim = predictor_config.token_dimensionality
      case 'prefix_simplex':
        prefix_type = 'simplex'
        prefix_dim = predictor_config.token_dimensionality
      case 'prefix_soft':
        prefix_type = 'embedding'
        prefix_dim = predictor_config.embedding_dimensionality
      case _:
        prefix_type = 'none'
        prefix_dim = 0

    for i in range(tuning_config.num_tuning_repetitions):
      logging.info(
          'Repetition %d out of %d', i + 1, tuning_config.num_tuning_repetitions
      )
      # Generate new initial prefix for each repetition.
      if tuning_config.tuning_method in prefix_tuning_methods:
        prefix_init_seed, prefix_init_seed_now = jrandom.split(prefix_init_seed)
        initial_prefix = generate_prefix(
            prefix_length=tuning_config.prefix_length,
            prefix_dimensionality=prefix_dim,
            prefix_init_method=tuning_config.prefix_init_method,
            rng_key=prefix_init_seed_now,
        )
      else:
        initial_prefix = None

      if tuning_config.iterate_datagen_seed_over_repetitions:
        datagen_seed += 1

      # ---- Tune ----
      tuned_params, tuned_prefix, tuning_results = tune(
          tuning_config=tuning_config,
          predictor_config=predictor_config,
          torso_config=torso_config,
          predictor_params=predictor_params,
          data_config=tuning_data_config,
          initial_prefix=initial_prefix,
          override_datagen_seed=datagen_seed,
      )
      if tuning_name not in results:
        results[tuning_name] = {'tuning_loss': [tuning_results['loss']]}
        if return_tuned_params:
          results[tuning_name]['tuned_params'] = ([tuned_params],)
        if return_tuned_prefix:
          results[tuning_name]['initial_prefix'] = [initial_prefix]
          results[tuning_name]['tuned_prefix'] = [tuned_prefix]
      else:
        results[tuning_name]['tuning_loss'].append(tuning_results['loss'])
        if return_tuned_params:
          results[tuning_name]['tuned_params'].append(tuned_params)
        if return_tuned_prefix:
          results[tuning_name]['initial_prefix'].append(initial_prefix)
          results[tuning_name]['tuned_prefix'].append(tuned_prefix)

      # Evaluate on all evaluation data generators given in eval_data_configs.
      for eval_name, eval_data_config in eval_data_configs.items():
        logging.info(
            'Evaluating tuned predictor on %s (%s sequences)',
            eval_data_config.generator_type,
            eval_data_config.batch_size,
        )
        # Generate sequences once per eval data generator.
        if eval_name not in sequences:
          data_generator = builders.build_datagen(eval_data_config)
          dg_results = data_generator.generate_solve_and_losses(
              rng_key=jrandom.PRNGKey(eval_datgen_seed)
          )
          sequences[eval_name] = dg_results[0]
          # Add Bayes-optimal and ground truth results.
          results['Bayes'][eval_name] = {
              'log_probs': [dg_results[1]],
              'losses': [dg_results[2]],
          }
          results['ground_truth'][eval_name] = {
              'log_probs': [dg_results[3]],
              'losses': [dg_results[4]],
          }

        # Evaluate tuned predictor.
        start_time = time.process_time()
        eval_results = evaluation.evaluate_predictor_from_sequences(
            predictor_config=predictor_config,
            torso_config=torso_config,
            predictor_params=tuned_params,
            sequences=sequences[eval_name],
            batch_size=eval_batching_batch_size,
            prefix_type=prefix_type,
            prefix=tuned_prefix,
        )
        logging.info(
            'Evaluation finished after %s seconds.',
            datetime.timedelta(seconds=time.process_time() - start_time),
        )
        if eval_name not in results[tuning_name]:
          results[tuning_name][eval_name] = {
              'log_probs': [eval_results[0]],
              'losses': [eval_results[1]],
          }
        else:
          results[tuning_name][eval_name]['log_probs'].append(eval_results[0])
          results[tuning_name][eval_name]['losses'].append(eval_results[1])

        # Optionally, and only once per eval_name, evaluate untuned predictor.
        if evaluate_untuned_predictor and eval_name not in results['NoTuning']:
          logging.info('Evaluating untuned predictor on %s.', eval_name)
          notuning_eval_results = evaluation.evaluate_predictor_from_sequences(
              predictor_config=predictor_config,
              torso_config=torso_config,
              predictor_params=predictor_params,
              sequences=sequences[eval_name],
              batch_size=eval_batching_batch_size,
              prefix_type='none',
              prefix=None,
          )
          results['NoTuning'][eval_name] = {
              'log_probs': [notuning_eval_results[0]],
              'losses': [notuning_eval_results[1]],
          }

  return results, sequences
