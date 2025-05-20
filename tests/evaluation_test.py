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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np

from thunnini.src import builders
from thunnini.src import config as config_lib
from thunnini.src import evaluation


_EVAL_DATA_CONFIG = config_lib.DirichletCategoricalGeneratorConfig(
    batch_size=8,
    sequence_length=16,
    vocab_size=2,
    alphas=np.array([1, 1]),
)

_PREDICTOR_CONFIG = config_lib.PredictorConfig(
    token_dimensionality=2,
    embedding_dimensionality=16,
)

_LINEAR_TORSO_CONFIG = config_lib.LinearTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
)

_TRANSFORMER_TORSO_CONFIG = config_lib.TransformerTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
    num_attention_heads=2,
    positional_encoding='SinCos',
    return_hidden_states=False,
    use_bias=False,
    widening_factor=4,
    normalize_qk=True,
    use_lora=True,
    reduced_rank=4,
)

_LSTM_TORSO_CONFIG = config_lib.LSTMTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
    return_hidden_states=False,
)


class EvaluateTest(parameterized.TestCase):

  @parameterized.parameters(
      [_LINEAR_TORSO_CONFIG, _LSTM_TORSO_CONFIG, _TRANSFORMER_TORSO_CONFIG]
  )
  def test_evaluation(self, torso_config: config_lib.PredictorTorsoConfig):
    data_generator = builders.build_datagen(_EVAL_DATA_CONFIG)
    dg_seed = 5
    # We need to split rng first to be compatible with test below that uses
    # the data generator internally (and splits the rng).
    data_rng = jax.random.split(jax.random.PRNGKey(dg_seed), 2)[0]
    batch_size = _EVAL_DATA_CONFIG.batch_size
    sequences = data_generator.generate(
        rng_key=data_rng, return_ground_truth_log_probs=False
    )

    predictor = builders.build_predictor(_PREDICTOR_CONFIG, torso_config)
    params = predictor.init(rngs=jax.random.PRNGKey(10), sequences=sequences)

    # ----- Test different evaluation functions -----
    with self.subTest('evaluate_from_sequences'):
      log_preds, losses = evaluation.evaluate_predictor_from_sequences(
          predictor_config=_PREDICTOR_CONFIG,
          torso_config=torso_config,
          predictor_params=params,
          sequences=sequences,
          batch_size=-1,
      )
      chex.assert_equal_shape([log_preds, sequences])
      chex.assert_shape(
          [losses], (batch_size, _EVAL_DATA_CONFIG.sequence_length)
      )
    with self.subTest('batched_eval'):
      log_preds_b, losses_b = evaluation.evaluate_predictor_from_sequences(
          predictor_config=_PREDICTOR_CONFIG,
          torso_config=torso_config,
          predictor_params=params,
          sequences=sequences,
          batch_size=int(batch_size / 2) + 1,  # Choose a wonky batch size.
      )
      chex.assert_trees_all_close(log_preds, log_preds_b, atol=1e-5)
      chex.assert_trees_all_close(losses, losses_b, rtol=1e-5)
    with self.subTest('datagen_eval'):
      sequences_d, log_preds_d, losses_d = (
          evaluation.evaluate_predictor_from_datagen(
              predictor_config=_PREDICTOR_CONFIG,
              torso_config=torso_config,
              predictor_params=params,
              datagen_config=_EVAL_DATA_CONFIG,
              datagen_seed=dg_seed,
              datagen_num_batches=2,  # Draw two batches for testing.
          )
      )
      chex.assert_shape(
          sequences_d,
          (
              2 * batch_size,
              _EVAL_DATA_CONFIG.sequence_length,
              _EVAL_DATA_CONFIG.vocab_size,
          ),
      )
      chex.assert_trees_all_close(sequences, sequences_d[:batch_size])
      chex.assert_trees_all_close(log_preds, log_preds_d[:batch_size])
      chex.assert_trees_all_close(losses, losses_d[:batch_size])


if __name__ == '__main__':
  absltest.main()
