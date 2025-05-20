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
import numpy as np

from thunnini.src import config as config_lib
from thunnini.src import training


_DATA_CONFIG = config_lib.DirichletCategoricalGeneratorConfig(
    batch_size=16,
    sequence_length=10,
    vocab_size=2,
    alphas=np.array([1, 1]),
)
_PREDICTOR_CONFIG = config_lib.PredictorConfig(
    token_dimensionality=2,
    embedding_dimensionality=16,
)
_TORSO_CONFIG = config_lib.LinearTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
)
_TRAINING_CONFIG = config_lib.TrainingConfig(
    num_training_steps=2,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    data_gen_seed=0,
    predictor_init_seed=0,
)


class TrainingTest(parameterized.TestCase):

  def test_training_loop(self):
    _, _ = training.train(
        training_config=_TRAINING_CONFIG,
        predictor_config=_PREDICTOR_CONFIG,
        torso_config=_TORSO_CONFIG,
        data_config=_DATA_CONFIG,
    )


if __name__ == '__main__':
  absltest.main()
