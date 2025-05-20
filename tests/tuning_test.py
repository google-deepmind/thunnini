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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np

from thunnini.src import builders
from thunnini.src import config as config_lib
from thunnini.src import tuning


_PREDICTOR_CONFIG = config_lib.PredictorConfig(
    token_dimensionality=2,
    embedding_dimensionality=16,
)

_LINEAR_TORSO_CONFIG = config_lib.LinearTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
)

_TUNING_CONFIG = config_lib.TuningConfig(
    num_tuning_steps=2,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    data_gen_seed=10,
    prefix_init_seed=10,
    tuning_method='prefix_real',
    prefix_length=5,
    prefix_init_method='one_hot',
    iterate_datagen_seed_over_repetitions=True,
)

_TUNING_DATA_CONFIG = config_lib.CategoricalGeneratorConfig(
    batch_size=16,
    sequence_length=10,
    vocab_size=2,
    biases=np.array([0.1, 0.9]),
)


class TuningTest(parameterized.TestCase):

  def test_tuning_fn(self):

    # Get some dummy predictor parameters
    predictor = builders.build_predictor(
        _PREDICTOR_CONFIG, _LINEAR_TORSO_CONFIG
    )
    data_generator = builders.build_datagen(_TUNING_DATA_CONFIG)
    sequences = data_generator.generate(
        rng_key=jax.random.PRNGKey(10), return_ground_truth_log_probs=False
    )
    params = predictor.init(rngs=jax.random.PRNGKey(815), sequences=sequences)

    with self.subTest('tune_with_real_prefix'):
      tuned_params, tuned_prefix, tuning_results = tuning.tune(
          tuning_config=_TUNING_CONFIG,
          predictor_config=_PREDICTOR_CONFIG,
          torso_config=_LINEAR_TORSO_CONFIG,
          predictor_params=params,
          data_config=_TUNING_DATA_CONFIG,
      )

    with self.subTest('prefix_tuning_does_not_modify_params'):
      chex.assert_trees_all_equal(tuned_params, params)

    with self.subTest('prefix_tuning_changes_prefix'):
      init_prefix = tuning_results['initial_prefix']
      self.assertGreater(np.sum(np.abs(tuned_prefix - init_prefix)), 0)

    with self.subTest('tune_full_parameters'):
      tuning_config = dataclasses.replace(
          _TUNING_CONFIG, tuning_method='full_parameters'
      )
      tuned_params, tuned_prefix, _ = tuning.tune(
          tuning_config=tuning_config,
          predictor_config=_PREDICTOR_CONFIG,
          torso_config=_LINEAR_TORSO_CONFIG,
          predictor_params=params,
          data_config=_TUNING_DATA_CONFIG,
      )

    with self.subTest('full_param_tuning_changes_params'):
      has_diff = jax.tree_util.tree_map(
          lambda a, b: not jax.numpy.allclose(a, b, rtol=1e-06),
          tuned_params,
          params,
      )
      self.assertTrue(all(jax.tree_util.tree_leaves(has_diff)))

    with self.subTest('full_param_tuning_does_not_produce_prefix'):
      self.assertIsNone(tuned_prefix)


if __name__ == '__main__':
  absltest.main()
