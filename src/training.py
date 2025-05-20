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

"""Implements predictor pretraining."""

import functools
from typing import Any

from absl import logging
from jax import random as jrandom
import optax

from thunnini.src import builders
from thunnini.src import config as config_lib
from thunnini.src import predictor_tuning_functions


def train(
    training_config: config_lib.TrainingConfig,
    predictor_config: config_lib.PredictorConfig,
    torso_config: config_lib.PredictorTorsoConfig,
    data_config: config_lib.DataGeneratorConfig,
) -> tuple[optax.Params, dict[str, Any]]:
  """Returns parameters and loss curve of model trained with given configs."""
  data_rng, data_init_rng = jrandom.split(
      jrandom.PRNGKey(training_config.data_gen_seed)
  )
  _, predictor_init_rng = jrandom.split(
      jrandom.PRNGKey(training_config.predictor_init_seed)
  )

  data_generator = builders.build_datagen(data_config)
  dummy_sequences = data_generator.generate(
      rng_key=data_init_rng, return_ground_truth_log_probs=False
  )

  predictor = builders.build_predictor(predictor_config, torso_config)
  params = predictor.init(
      rngs=predictor_init_rng,
      sequences=dummy_sequences,
  )

  if not torso_config.is_trainable:
    logging.info('Skipping training since the torso is not trainable.')
    return params, {}

  optimizer = optax.chain(
      optax.clip_by_global_norm(training_config.max_grad_norm),
      optax.adam(training_config.learning_rate),
  )
  opt_state = optimizer.init(params=params)
  grad_fn = predictor_tuning_functions.make_grad_fn(
      predictor=predictor, tuning_method='full_parameters'
  )
  update_fn = functools.partial(
      predictor_tuning_functions.update_parameters,
      grad_fn=grad_fn,
      optimizer=optimizer,
      prefix_type='none',
      prefix=None,
      tuning_method='full_parameters',
  )

  results = {
      'loss': [],
      'grad_norm_unclipped': [],
  }
  for _ in range(0, training_config.num_training_steps):
    data_rng, data_rng_now = jrandom.split(data_rng)
    sequences = data_generator.generate(
        rng_key=data_rng_now, return_ground_truth_log_probs=False
    )

    params, _, opt_state, loss, grad_norm_unclipped = update_fn(
        params=params,
        opt_state=opt_state,
        sequences=sequences,
    )

    results['loss'].append(loss)
    results['grad_norm_unclipped'].append(grad_norm_unclipped)

  return params, results
