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

"""Predictor torso implementations.

Add your own torso by subclassing `predictor.PredictorTorso` and implement the
`__call__` method. The rest is handled by the `predictor.Predictor` wrapper.

Do not forget to add a config in `config.py`, and include the torso in all the
relevant test files. Also add your torso to all torsos in `builders.py` and
`types.py`.
"""

from flax import linen as nn
from jax import numpy as jnp
from jax import random as jrandom
import jaxtyping as jtp

from thunnini.src import config as config_lib
from thunnini.src import predictor
from thunnini.src import transformer_utils
from thunnini.src import types


RecurrentState = jtp.Float32[jtp.Array, '2 B H']
RecurrentOutput = jtp.Float32[jtp.Array, 'B O']


class LinearPredictorTorso(predictor.PredictorTorso):
  """Linear predictor torso. Mainly used for testing."""

  config: config_lib.LinearTorsoConfig

  @nn.compact
  def __call__(
      self,
      embeddings: types.Embeddings,
  ) -> tuple[types.TorsoOutputs, types.Hidden]:
    """Returns torso output activations for embeddings."""
    output = embeddings
    for hidden_size in self.config.hidden_sizes:
      output = nn.Dense(hidden_size)(output)
    if self.config.return_hidden_states:
      raise ValueError('Linear torso has no hidden states.')
    else:
      states = None
    return output, states


class TransformerPredictorTorso(predictor.PredictorTorso):
  """Transformer (decoder) predictor torso.

  This multi-layer, multi-head attention decoder consists of one layer per entry
  in `config.hidden_sizes`. The width or feature dimension of all parts is equal
  to the embedding dimensionality, except the first dense layer of the MLP block
  which has dimensionality `hidden_size * config.widening_factor`. LoRA adapters
  can be enabled by setting `config.use_lora` to True and specifying the reduced
  rank in `config.reduced_rank`. LoRA adapters are in parallel to all dense
  layers in the attention block and initialized such that they have no
  contribution to the output unless they are finetuned.
  """

  config: config_lib.TransformerTorsoConfig

  @nn.compact
  def __call__(
      self,
      embeddings: types.Embeddings,
  ) -> tuple[types.TorsoOutputs, types.Hidden]:
    """Returns torso output activations for embeddings."""
    outputs = transformer_utils.add_positional_encodings(
        embeddings, self.config.positional_encoding
    )

    hidden_states = {} if self.config.return_hidden_states else None
    for layer, hidden_size in enumerate(self.config.hidden_sizes):
      attention_input = nn.LayerNorm()(outputs)
      attention = transformer_utils.attention_block(
          attention_input,
          num_heads=self.config.num_attention_heads,
          normalize_qk=self.config.normalize_qk,
          use_bias=self.config.use_bias,
          use_lora=self.config.use_lora,
          reduced_rank=self.config.reduced_rank,
      )
      outputs += attention
      states = attention

      mlp_input = nn.LayerNorm()(outputs)
      mlp_output = transformer_utils.mlp_block(
          mlp_input,
          hidden_size=hidden_size,
          widening_factor=self.config.widening_factor,
          use_bias=self.config.use_bias,
      )
      outputs += mlp_output
      outputs = nn.LayerNorm()(outputs)

      if hidden_states is not None:
        hidden_states[f'layer{layer}_attention_out'] = jnp.array(states)

    return outputs, hidden_states


class LSTMPredictorTorso(predictor.PredictorTorso):
  """LSTM predictor torso."""

  config: config_lib.LSTMTorsoConfig

  @nn.compact
  def __call__(
      self,
      embeddings: types.Embeddings,
  ) -> tuple[types.TorsoOutputs, types.Hidden]:
    """Returns torso output activations for embeddings."""
    hidden_sizes = list(self.config.hidden_sizes)
    batch_size = embeddings.shape[0]
    cells = []
    cell_init_rngs = jrandom.split(jrandom.PRNGKey(1), len(hidden_sizes))
    input_width_per_layer = [embeddings.shape[-1]] + hidden_sizes[:-1]
    initial_states = []

    # Iterate through layers and initialize cells.
    for layer, hidden_size in enumerate(hidden_sizes):
      cells.append(nn.OptimizedLSTMCell(hidden_size))
      input_shape = (batch_size, input_width_per_layer[layer])
      initial_states.append(
          cells[layer].initialize_carry(
              rng=cell_init_rngs[layer], input_shape=input_shape
          )
      )

    # We need to create a wrapper since flax.linen.scan does not support
    # `return_all_states`, unlike `hk.dynamic_unroll`.
    def unroll_fn(
        cell: nn.RNNCellBase,
        state: RecurrentState,
        inputs: types.Embeddings,
    ) -> tuple[RecurrentState, tuple[RecurrentState, RecurrentOutput]]:
      """Unroll function for a single layer."""
      state, output = cell(state, inputs)
      return state, (state, output)

    scan_layer = nn.scan(
        target=unroll_fn,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=-2,  # Scan over the time-step dimension.
        out_axes=-2,  # Collect results over the time-step dimension.
    )

    hidden_states = {} if self.config.return_hidden_states else None
    outputs = embeddings
    for layer, (cell, initial_state) in enumerate(zip(cells, initial_states)):
      _, (states, outputs) = scan_layer(cell, initial_state, outputs)
      if hidden_states is not None:
        hidden_states[f'layer{layer}_cell'] = jnp.array(states[0])
        hidden_states[f'layer{layer}_hidden'] = jnp.array(states[1])

    return outputs, hidden_states
