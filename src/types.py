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

"""Type definitions for readability and static type checks."""

from typing import Literal

import jaxtyping as jtp

# B ... Batch size.
# T ... Sequence length.
# V ... Vocabulary size.
# K ... Prefix length.
# E ... Embedding size.
# O ... Output size of torso.
# H ... Hidden size (may differ per layer).
Sequences = jtp.UInt32[jtp.Array, 'B T V']
LogPredictions = jtp.Float32[jtp.Array, 'B T V']
Losses = jtp.Float32[jtp.Array, 'B T 1']
Embeddings = jtp.Float32[jtp.Array, 'B K+T E']
PrefixPrompt = jtp.Float32[jtp.Array, 'K V']
PrefixLogPredictions = jtp.Float32[jtp.Array, 'K V'] | None
PrefixEmbedding = jtp.Float32[jtp.Array, 'K E']
TorsoOutputs = jtp.Float32[jtp.Array, 'B K+T O']

LayerAndParamName = str  # E.g., 'layer0_cell' or 'layer1_hidden'.
TorsoHidden = dict[LayerAndParamName, jtp.Float32[jtp.Array, 'B K+T H']] | None
Hidden = dict[LayerAndParamName, jtp.Float32[jtp.Array, 'B T H']] | None
PrefixHidden = dict[LayerAndParamName, jtp.Float32[jtp.Array, 'B K H']] | None

TuningMethodType = Literal[
    'prefix_real',  # Real-valued prompt prefix (no constraints).
    'prefix_simplex',  # Real-valued prompt prefix that lies in simplex spanned
    # by the (one-hot) tokens. Passed through softmax.
    'prefix_soft',  # Prefix in embedding space.
    'full_parameters',  # Fine tuning of all model parameters (except LoRA).
    'lora_finetune',  # Low-rank fine tuning.
    'embedding',  # Fine tune only the embedding layer.
    'unembedding',  # Fine tune only the unembedding layer.
    'embedding_unembedding',  # Fine tune embedding and unembedding layers.
]

PrefixType = Literal[
    'none',  # No prefix.
    'prepend',  # Real-valued or hard-token prompt prefix - simply prepend.
    'simplex',  # Raw prefix is passed through softmax first.
    'embedding',  # Prefix is for embedding space (soft-prompt).
]

PrefixInitMethod = Literal[
    'zeros',  # Initialize prefix with zeros.
    'simplex',  # Initialize prefix with random values in [0, 1].
    'one_hot',  # Initialize prefix with random one-hot values.
]

Prefix = PrefixPrompt | PrefixEmbedding | None

DataGenType = Literal[
    'Dirichlet-Categorical', 'Categorical', 'Mixture-of-Categoricals'
]

TorsoType = Literal['Linear', 'Transformer', 'LSTM']

PositionalEncodingType = Literal['SinCos']
