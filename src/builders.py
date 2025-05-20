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

"""Builders for data generators and torsos."""

from collections.abc import Callable
import typing
from typing import Any

from thunnini.src import config as config_lib
from thunnini.src import data_generator_base
from thunnini.src import data_generators
from thunnini.src import predictor
from thunnini.src import predictor_torsos
from thunnini.src import types

ALL_DATAGENS: dict[
    str,
    Callable[..., data_generator_base.DataGenerator],
] = {
    'Dirichlet-Categorical': data_generators.DirichletCategoricalGenerator,
    'Categorical': data_generators.CategoricalGenerator,
    'Mixture-of-Categoricals': data_generators.MixtureOfCategoricalsGenerator,
}

ALL_DATAGEN_CONFIGS: dict[
    str,
    Callable[..., config_lib.DataGeneratorConfig],
] = {
    'Dirichlet-Categorical': config_lib.DirichletCategoricalGeneratorConfig,
    'Categorical': config_lib.CategoricalGeneratorConfig,
    'Mixture-of-Categoricals': config_lib.MixtureOfCategoricalsGeneratorConfig,
}

ALL_TORSOS: dict[
    str,
    Callable[..., predictor.PredictorTorso],
] = {
    'Linear': predictor_torsos.LinearPredictorTorso,
    'Transformer': predictor_torsos.TransformerPredictorTorso,
    'LSTM': predictor_torsos.LSTMPredictorTorso,
}

ALL_TORSO_CONFIGS: dict[
    str,
    Callable[..., config_lib.PredictorTorsoConfig],
] = {
    'Linear': config_lib.LinearTorsoConfig,
    'Transformer': config_lib.TransformerTorsoConfig,
    'LSTM': config_lib.LSTMTorsoConfig,
}


def build_datagen(
    config: config_lib.DataGeneratorConfig | dict[str, Any],
) -> data_generator_base.DataGenerator:
  """Returns a data generator for the given config."""
  if isinstance(config, dict):
    config = config_lib.DataGeneratorConfig(**config)
  assert config.generator_type in typing.get_args(types.DataGenType)
  return ALL_DATAGENS[config.generator_type](config)


def get_torso_builder(
    config: config_lib.PredictorTorsoConfig | dict[str, Any],
) -> Callable[..., predictor.PredictorTorso]:
  """Returns a torso builder for the given config."""
  if isinstance(config, dict):
    config = ALL_TORSO_CONFIGS[config['torso_type']](**config)
  assert config.torso_type in typing.get_args(types.TorsoType)
  return ALL_TORSOS[config.torso_type]


def build_predictor(
    predictor_config: config_lib.PredictorConfig | dict[str, Any],
    torso_config: config_lib.PredictorTorsoConfig | dict[str, Any],
) -> predictor.Predictor:
  """Returns a predictor with torso as specified by the configs."""
  if isinstance(predictor_config, dict):
    predictor_config = config_lib.PredictorConfig(**predictor_config)
  if isinstance(torso_config, dict):
    torso_config = ALL_TORSO_CONFIGS[torso_config['torso_type']](**torso_config)
  torso_builder = get_torso_builder(torso_config)
  return predictor.Predictor(
      config=predictor_config,
      torso_config=torso_config,
      torso_builder=torso_builder,
  )
