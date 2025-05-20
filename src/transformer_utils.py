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

"""Utility functions for the transformer torso."""

from flax import linen as nn
from jax import numpy as jnp
import jaxtyping as jtp

from thunnini.src import types


def sinusoid_positional_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> jnp.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D].
  """
  freqs = jnp.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  pos_seq = jnp.arange(start=0, stop=sequence_length)

  sinusoid_inp = jnp.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = jnp.concatenate(
      [jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]


def add_positional_encodings(
    embeddings: types.Embeddings,
    pos_encoding_type: types.PositionalEncodingType,
) -> types.Embeddings:
  """Adds positional encodings to the embeddings."""

  if pos_encoding_type == 'SinCos':
    pos_encodings = sinusoid_positional_encoding(
        sequence_length=embeddings.shape[1],
        hidden_size=embeddings.shape[-1],
    )
  else:
    raise ValueError(f'Unsupported positional encoding: {pos_encoding_type}')

  return embeddings + pos_encodings


def mlp_block(
    inputs: types.Embeddings,
    hidden_size: int,
    widening_factor: int,
    use_bias: bool,
) -> types.Embeddings:
  """MLP block for the Transformer."""
  input_size = inputs.shape[-1]
  widened_size = hidden_size * widening_factor
  out = nn.Dense(features=widened_size, use_bias=use_bias)(inputs)
  out = nn.gelu(out)
  out = nn.Dense(features=input_size, use_bias=use_bias)(out)
  return out


def attention_block(
    inputs: types.Embeddings,
    num_heads: int,
    normalize_qk: bool = True,
    use_bias: bool = False,
    use_lora: bool = False,
    reduced_rank: int = 1,
) -> types.Embeddings:
  """Attention block for the Transformer."""
  causal_mask = nn.make_causal_mask(inputs[:, :, 0])

  attn = MultiheadAttention(
      num_heads=num_heads,
      normalize_qk=normalize_qk,
      use_bias=use_bias,
      use_lora=use_lora,
      reduced_rank=reduced_rank,
  )
  return attn(
      inputs_q=inputs,
      inputs_kv=inputs,
      mask=causal_mask,
  )


class DenseWithLora(nn.Module):
  """Dense layer with optional LoRA."""

  features: int
  use_bias: bool
  use_lora: bool = False
  reduced_rank: int = 1

  @nn.compact
  def __call__(self, inputs: types.Embeddings) -> types.Embeddings:
    full_rank = nn.Dense(
        features=self.features,
        use_bias=self.use_bias,
        name=self.name,
    )
    out = full_rank(inputs)

    if self.use_lora:
      lora_in = nn.Dense(
          features=self.reduced_rank,
          use_bias=False,  # lora_in bias is redundant if lora_out has bias.
          name='LoRA_in_' + self.name,
      )
      lora_out = nn.Dense(
          features=self.features,
          use_bias=self.use_bias,
          name='LoRA_out_' + self.name,
          kernel_init=nn.initializers.zeros_init(),
          bias_init=nn.initializers.zeros_init(),
      )
      lora_act = lora_in(inputs)
      lora_act = lora_out(lora_act)
      out = out + lora_act
    return out


class MultiheadAttention(nn.Module):
  """Multihead attention."""

  num_heads: int
  use_bias: bool = False
  normalize_qk: bool = True
  use_lora: bool = False
  reduced_rank: int = 1

  @nn.compact
  def __call__(
      self,
      inputs_q: types.Embeddings,
      inputs_kv: types.Embeddings,
      mask: jtp.Array,
  ):
    batch_size, sequence_length, input_size = inputs_q.shape
    query = DenseWithLora(
        features=input_size,
        use_bias=self.use_bias,
        name='query',
        use_lora=self.use_lora,
        reduced_rank=self.reduced_rank,
    )(inputs_q)
    key = DenseWithLora(
        features=input_size,
        use_bias=self.use_bias,
        name='key',
        use_lora=self.use_lora,
        reduced_rank=self.reduced_rank,
    )(inputs_kv)
    value = DenseWithLora(
        features=input_size,
        use_bias=self.use_bias,
        name='value',
        use_lora=self.use_lora,
        reduced_rank=self.reduced_rank,
    )(inputs_kv)

    # Layer norm
    if self.normalize_qk:
      query = nn.LayerNorm(use_bias=False)(query)
      key = nn.LayerNorm(use_bias=False)(key)

    assert input_size % self.num_heads == 0, (
        f'Embedding dimension ({input_size}) must be divisible by number of'
        f' heads ({self.num_heads}).'
    )
    head_dim = input_size // self.num_heads
    # Split per head
    query = query.reshape(batch_size, sequence_length, self.num_heads, head_dim)
    key = key.reshape(batch_size, sequence_length, self.num_heads, head_dim)
    value = value.reshape(batch_size, sequence_length, self.num_heads, head_dim)

    logits = nn.dot_product_attention(query, key, value, mask=mask)

    # Concat across heads
    logits = logits.reshape(batch_size, sequence_length, input_size)

    # Attention weights
    logits = DenseWithLora(
        features=input_size,
        use_bias=self.use_bias,
        name='attention_weights',
        use_lora=self.use_lora,
        reduced_rank=self.reduced_rank,
    )(logits)

    return logits
