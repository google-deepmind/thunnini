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

"""Predictor wrapper class, adds un/embedding and prefix prompting to torso."""

import abc
from collections.abc import Callable

import chex
import flax.linen as nn
import jax
from jax import numpy as jnp

from thunnini.src import config as config_lib
from thunnini.src import types


class PredictorTorso(nn.Module, abc.ABC):
  """Base class for predictor torso (sits between embedding and unembedding)."""

  config: config_lib.PredictorTorsoConfig

  @abc.abstractmethod
  def __call__(
      self,
      embeddings: types.Embeddings,
  ) -> tuple[types.TorsoOutputs, types.TorsoHidden]:
    """Returns torso output activations and states for embeddings.

    Args:
      embeddings: Batch of embeddings to process.

    Returns:
      Outputs and hidden (internal) states of predictor torso.
    """


class Predictor(nn.Module):
  """Wrapper class for all predictors.

  Takes a predictor torso and wraps it with an embedding and unembedding, and
  the functionality required for the various prefix prompting and tuning
  methods.

  To implement additional predictors, preferrably implement a new torso, as this
  class ensures compatibility with all functionality of the library.
  """

  config: config_lib.PredictorConfig
  torso_config: config_lib.PredictorTorsoConfig
  torso_builder: Callable[..., PredictorTorso]

  @nn.compact
  def __call__(
      self,
      sequences: types.Sequences,
      prefix_type: types.PrefixType = 'none',
      prefix: types.Prefix = None,
  ) -> tuple[
      types.LogPredictions,
      types.Hidden,
      types.PrefixLogPredictions,
      types.PrefixHidden,
  ]:
    """Returns log predictions for sequences with optional prefix.

    Args:
      sequences: Batch of one-hot sequences to predict.
      prefix_type: The type of the prefix. Can be 'none' for no prefix.
      prefix: The prefix that is prepended to sequences.

    Returns:
      logits: Log predictions for sequences. The first prediction is made before
        "seeing" the first token; the final prediction is for the final
        token(`sequences[:, -1, :]`).
      states: Hidden torso states for sequences (may be None, depending on torso
        config).
      prefix_logits: Log predictions for the prefix.
      prefix_states: Hidden torso states for the prefix (may be None, depending
        on torso config).
    """
    if prefix_type == 'none' and prefix is not None:
      raise ValueError(
          'Non-empty prefix prompt passed, but prefix prompt type is "none".'
      )
    batch_size, sequence_length, vocab_size = sequences.shape
    if prefix is not None:
      prefix_length = prefix.shape[0]
    else:
      prefix_length = 0

    prefixed_sequences = sequences
    match prefix_type:
      case 'none':
        pass
      case 'prepend':
        prefixed_sequences = jnp.concatenate(
            (jnp.tile(prefix, (batch_size, 1, 1)), sequences),
            axis=1,
        )
      case 'simplex':
        prefix = jax.nn.softmax(prefix)
        prefixed_sequences = jnp.concatenate(
            (jnp.tile(prefix, (batch_size, 1, 1)), sequences),
            axis=1,
        )
      case 'embedding':
        pass  # We deal with this case after computing the embeddings.

    # Since we also want a prediction before having seen the first token, we
    # prepend a constant vector of zeros.
    sequences_w_bos = jnp.pad(
        prefixed_sequences,
        ((0, 0), (1, 0), (0, 0)),
        mode='constant',
        constant_values=0,
    )

    embeddings = nn.Dense(
        self.config.embedding_dimensionality, use_bias=True, name='embedding'
    )(sequences_w_bos)

    if prefix_type == 'embedding':
      # Prepend the prefix embedding to the embeddings, making sure, that the
      # bos embeddings (from initial zero vector) stay at the beginning.
      embeddings = jnp.concatenate(
          (
              embeddings[:, 0:1, :],  # Use slicing to keep dims.
              jnp.tile(prefix, (batch_size, 1, 1)),
              embeddings[:, 1:, :],
          ),
          axis=1,
      )

    torso = self.torso_builder(config=self.torso_config)
    torso_activations, torso_states = torso(embeddings)

    unembeddings = nn.Dense(vocab_size, use_bias=True, name='unembedding')(
        torso_activations
    )

    # Remove parts from activations and states that correspond to the prefix and
    # the final prediction (for the next input at the end sequences).
    logits = unembeddings[:, prefix_length:-1, :]
    chex.assert_trees_all_equal_shapes(logits, sequences)
    hidden_sizes = list(self.torso_config.hidden_sizes)
    if torso_states is not None:
      states = {}
      for k, v in torso_states.items():
        states[k] = v[:, prefix_length:-1, :]
        layer, _ = k.split('_', 1)
        layer = int(layer.removeprefix('layer'))
        if isinstance(self.torso_config, config_lib.TransformerTorsoConfig):
          target_size = self.config.embedding_dimensionality
        else:
          target_size = hidden_sizes[layer]
        chex.assert_tree_shape_prefix(
            states[k], (batch_size, sequence_length, target_size)
        )
    else:
      states = None

    if prefix_length > 0:
      prefix_logits = unembeddings[:, :prefix_length, :]
      chex.assert_shape(prefix_logits, (batch_size, prefix_length, vocab_size))
      if torso_states is not None:
        prefix_states = {}
        for k, v in torso_states.items():
          prefix_states[k] = v[:, :prefix_length, :]
          layer, _ = k.split('_', 1)
          layer = int(layer.removeprefix('layer'))
          if isinstance(self.torso_config, config_lib.TransformerTorsoConfig):
            target_size = self.config.embedding_dimensionality
          else:
            target_size = hidden_sizes[layer]
          chex.assert_tree_shape_prefix(
              prefix_states[k], (batch_size, prefix_length, target_size)
          )
      else:
        prefix_states = None
    else:
      prefix_logits = None
      prefix_states = None

    return logits, states, prefix_logits, prefix_states
