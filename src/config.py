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

"""Configuration (dataclasses) for tuning library.

The main design principle is that configs can be serialized into plain text (via
first converting to dicts), and can be re-instantiated from text (by first
converting back into a dict). This requires no nesting of dataclasses, and no
post-instantiation manipulation of default values.
"""

from collections.abc import Iterable
import dataclasses

import jaxtyping as jtp

from thunnini.src import types


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataGeneratorConfig:
  generator_type: types.DataGenType
  batch_size: int
  sequence_length: int
  vocab_size: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class DirichletCategoricalGeneratorConfig(DataGeneratorConfig):
  generator_type: types.DataGenType = 'Dirichlet-Categorical'
  alphas: jtp.Float32[jtp.Array, 'V']


@dataclasses.dataclass(frozen=True, kw_only=True)
class CategoricalGeneratorConfig(DataGeneratorConfig):
  generator_type: types.DataGenType = 'Categorical'
  biases: jtp.Float32[jtp.Array, 'V']


@dataclasses.dataclass(frozen=True, kw_only=True)
class MixtureOfCategoricalsGeneratorConfig(DataGeneratorConfig):
  generator_type: types.DataGenType = 'Mixture-of-Categoricals'
  mixing_weights: jtp.Float32[jtp.Array, 'C']
  biases: jtp.Float32[jtp.Array, 'C V']


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainingConfig:
  num_training_steps: int
  learning_rate: float
  max_grad_norm: float
  data_gen_seed: int
  predictor_init_seed: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class TuningConfig:
  num_tuning_steps: int
  learning_rate: float
  max_grad_norm: float
  data_gen_seed: int
  prefix_init_seed: int
  tuning_method: types.TuningMethodType
  num_tuning_repetitions: int = 1
  prefix_length: int | None = None
  prefix_init_method: types.PrefixInitMethod | None = 'simplex'
  iterate_datagen_seed_over_repetitions: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class PredictorConfig:
  token_dimensionality: int  # Dimensionality of inputs.
  embedding_dimensionality: int  # Dimensionality after applying embedding.


@dataclasses.dataclass(frozen=True, kw_only=True)
class PredictorTorsoConfig:
  torso_type: types.TorsoType
  is_trainable: bool
  hidden_sizes: Iterable[int]  # One entry per layer.
  return_hidden_states: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearTorsoConfig(PredictorTorsoConfig):
  torso_type: types.TorsoType = 'Linear'


@dataclasses.dataclass(frozen=True, kw_only=True)
class TransformerTorsoConfig(PredictorTorsoConfig):
  torso_type: types.TorsoType = 'Transformer'
  num_attention_heads: int
  positional_encoding: types.PositionalEncodingType = 'SinCos'
  widening_factor: int = 4  # Hidden size of first dense layer of MLP block in
  # transformer layer is multiplied by this factor.
  normalize_qk: bool = True  # Whether to use layer norm for q and k.
  use_bias: bool = False  # Whether to use bias in all dense layers.
  use_lora: bool = False  # If true, all dense layers of the attention block
  # have an additional LoRA block with the given reduced rank in parallel.
  reduced_rank: int = 4  # Only relevant if use_lora is True.


@dataclasses.dataclass(frozen=True, kw_only=True)
class LSTMTorsoConfig(PredictorTorsoConfig):
  torso_type: types.TorsoType = 'LSTM'
