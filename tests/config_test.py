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

from collections.abc import Callable
import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np

from thunnini.src import config as config_lib


AllConfigTypes = (
    config_lib.DataGeneratorConfig
    | config_lib.PredictorConfig
    | config_lib.PredictorTorsoConfig
    | config_lib.TrainingConfig
)

# ----- Data generator configs -----
CATEGORICAL_CONFIG = config_lib.CategoricalGeneratorConfig(
    batch_size=128,
    sequence_length=100,
    vocab_size=2,
    biases=np.array([0.2, 0.8]),
)
CATEGORICAL_CONFIG_DICT = {
    'generator_type': 'Categorical',
    'batch_size': 128,
    'sequence_length': 100,
    'vocab_size': 2,
    'biases': np.array([0.2, 0.8]),
}
CATEGORICAL_MIXTURE_CONFIG = config_lib.MixtureOfCategoricalsGeneratorConfig(
    batch_size=128,
    sequence_length=100,
    vocab_size=2,
    mixing_weights=np.array([0.5, 0.5]),
    biases=np.array([[0.2, 0.8], [0.8, 0.2]]),
)
CATEGORICAL_MIXTURE_CONFIG_DICT = {
    'generator_type': 'Mixture-of-Categoricals',
    'batch_size': 128,
    'sequence_length': 100,
    'vocab_size': 2,
    'mixing_weights': np.array([0.5, 0.5]),
    'biases': np.array([[0.2, 0.8], [0.8, 0.2]]),
}
DIRICHLET_CONFIG = config_lib.DirichletCategoricalGeneratorConfig(
    batch_size=128,
    sequence_length=100,
    vocab_size=2,
    alphas=np.array([1, 1]),
)
DIRICHLET_CONFIG_DICT = {
    'generator_type': 'Dirichlet-Categorical',
    'batch_size': 128,
    'sequence_length': 100,
    'vocab_size': 2,
    'alphas': np.array([1, 1]),
}

# ----- Training config -----
TRAINING_CONFIG = config_lib.TrainingConfig(
    num_training_steps=1000,
    learning_rate=0.001,
    max_grad_norm=1.0,
    data_gen_seed=0,
    predictor_init_seed=0,
)
TRAINING_CONFIG_DICT = {
    'num_training_steps': 1000,
    'learning_rate': 0.001,
    'max_grad_norm': 1.0,
    'data_gen_seed': 0,
    'predictor_init_seed': 0,
}

# ----- Tuning config -----
TUNING_CONFIG = config_lib.TuningConfig(
    num_tuning_steps=1000,
    learning_rate=0.001,
    max_grad_norm=1.0,
    data_gen_seed=0,
    prefix_init_seed=0,
    tuning_method='prefix_real',
    num_tuning_repetitions=1,
    prefix_length=6,
    prefix_init_method='simplex',
    iterate_datagen_seed_over_repetitions=True,
)
TUNING_CONFIG_DICT = {
    'num_tuning_steps': 1000,
    'learning_rate': 0.001,
    'max_grad_norm': 1.0,
    'data_gen_seed': 0,
    'prefix_init_seed': 0,
    'tuning_method': 'prefix_real',
    'num_tuning_repetitions': 1,
    'prefix_length': 6,
    'prefix_init_method': 'simplex',
    'iterate_datagen_seed_over_repetitions': True,
}

# ----- Predictor config -----
PREDICTOR_CONFIG = config_lib.PredictorConfig(
    token_dimensionality=2,
    embedding_dimensionality=128,
)
PREDICTOR_CONFIG_DICT = {
    'token_dimensionality': 2,
    'embedding_dimensionality': 128,
}

# ----- Torso configs -----
LINEAR_TORSO_CONFIG = config_lib.LinearTorsoConfig(
    is_trainable=True,
    hidden_sizes=[128, 64],
    return_hidden_states=False,
)
LINEAR_TORSO_CONFIG_DICT = {
    'torso_type': 'Linear',
    'is_trainable': True,
    'hidden_sizes': [128, 64],
    'return_hidden_states': False,
}
TRANSFORMER_TORSO_CONFIG = config_lib.TransformerTorsoConfig(
    is_trainable=True,
    hidden_sizes=[128, 64],
    num_attention_heads=4,
    positional_encoding='SinCos',
    return_hidden_states=False,
    use_bias=False,
    widening_factor=4,
    normalize_qk=True,
    use_lora=False,
    reduced_rank=4,
)
TRANSFORMER_TORSO_CONFIG_DICT = {
    'torso_type': 'Transformer',
    'is_trainable': True,
    'hidden_sizes': [128, 64],
    'num_attention_heads': 4,
    'positional_encoding': 'SinCos',
    'return_hidden_states': False,
    'use_bias': False,
    'widening_factor': 4,
    'normalize_qk': True,
    'use_lora': False,
    'reduced_rank': 4,
}
LSTM_TORSO_CONFIG = config_lib.LSTMTorsoConfig(
    is_trainable=True,
    hidden_sizes=[128, 64],
    return_hidden_states=False,
)
LSTM_TORSO_CONFIG_DICT = {
    'torso_type': 'LSTM',
    'is_trainable': True,
    'hidden_sizes': [128, 64],
    'return_hidden_states': False,
}


class ConfigRebuildingTest(parameterized.TestCase):
  # To add a new config to the test, add the config object and corresponding
  # dict above, and in the list of parameters for this test add a tuple of
  # (config_object, config_dict, config_instantiation_fn).

  @parameterized.parameters([
      [
          CATEGORICAL_CONFIG,
          CATEGORICAL_CONFIG_DICT,
          config_lib.CategoricalGeneratorConfig,
      ],
      [
          CATEGORICAL_MIXTURE_CONFIG,
          CATEGORICAL_MIXTURE_CONFIG_DICT,
          config_lib.MixtureOfCategoricalsGeneratorConfig,
      ],
      [
          DIRICHLET_CONFIG,
          DIRICHLET_CONFIG_DICT,
          config_lib.DirichletCategoricalGeneratorConfig,
      ],
      [TRAINING_CONFIG, TRAINING_CONFIG_DICT, config_lib.TrainingConfig],
      [PREDICTOR_CONFIG, PREDICTOR_CONFIG_DICT, config_lib.PredictorConfig],
      [
          LINEAR_TORSO_CONFIG,
          LINEAR_TORSO_CONFIG_DICT,
          config_lib.LinearTorsoConfig,
      ],
      [
          TRANSFORMER_TORSO_CONFIG,
          TRANSFORMER_TORSO_CONFIG_DICT,
          config_lib.TransformerTorsoConfig,
      ],
      [LSTM_TORSO_CONFIG, LSTM_TORSO_CONFIG_DICT, config_lib.LSTMTorsoConfig],
  ])
  def test_config_instantiation(
      self,
      config: AllConfigTypes,
      config_dict: dict[str, Any],
      config_instanatiation_fn: Callable[..., AllConfigTypes],
  ):
    """Test that configs can be instantiated from dicts."""
    config_from_dict = config_instanatiation_fn(**config_dict)
    dict_from_orig_config = dataclasses.asdict(config)
    dict_from_rebuilt_config = dataclasses.asdict(config_from_dict)
    with self.subTest('configs_asdict_matches'):
      # Make sure that turning orig. config and config built from dict back into
      # dicts results in the same entries.
      chex.assert_trees_all_equal(
          dict_from_orig_config, dict_from_rebuilt_config
      )
    with self.subTest('rebuilt_config_matches_dict'):
      # Also make sure that we recover the original dict.
      chex.assert_trees_all_equal(dict_from_rebuilt_config, config_dict)


if __name__ == '__main__':
  absltest.main()
