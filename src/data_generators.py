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

"""Data generators for tuning library.

Add your own generator by subclassing `data_generator_base.DataGenerator` and
adding a config to `config.py`. Implement the `_generate` and `_solve` methods
in your subclass - the rest is handled by the base class.

Do not forget to add tests to `data_generators_test.py` and to add your
generator to all generators in `builders.py` and `types.py`.
"""

import chex
import distrax
from jax import numpy as jnp
from jax import random as jrandom
from jax import scipy as jsp
import numpy as np

from thunnini.src import config as config_lib
from thunnini.src import data_generator_base
from thunnini.src import types


class DirichletCategoricalGenerator(data_generator_base.DataGenerator):
  """Categorical from Dirichlet, then sequence i.i.d. from categorical."""

  def __init__(
      self, config: config_lib.DirichletCategoricalGeneratorConfig
  ) -> None:
    chex.assert_shape(config.alphas, (config.vocab_size,))
    self.alphas = config.alphas
    self.prior = distrax.Dirichlet(config.alphas.astype(jnp.float32))
    super().__init__(config)

  def _generate(
      self,
      rng_key: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[types.Sequences, types.LogPredictions]:
    prior_rng, data_rng = jrandom.split(rng_key, num=2)
    probs = self.prior.sample(seed=prior_rng, sample_shape=(batch_size))
    samples = (
        distrax.OneHotCategorical(probs=probs)
        .sample(seed=data_rng, sample_shape=(seq_length))
        .astype(jnp.float32)
    )
    samples = jnp.transpose(samples, axes=(1, 0, 2))
    gt_log_probs = jnp.tile(
        jnp.log(probs)[:, jnp.newaxis, :], [1, seq_length, 1]
    )
    return samples, gt_log_probs

  def _solve(self, sequences: types.Sequences) -> types.LogPredictions:
    batch_size, _, vocab_size = sequences.shape
    # Count "heads and tails".
    ht_counts = jnp.cumsum(sequences, axis=1)
    # Prepend zeros (counts before having seen any data).
    ht_counts = jnp.concatenate(
        [jnp.zeros((batch_size, 1, vocab_size)), ht_counts], axis=1
    )
    # Add virtual counts from prior.
    ht_counts = ht_counts + self.alphas
    # Normalize.
    log_probs = jnp.log(ht_counts) - jnp.log(ht_counts.sum(-1, keepdims=True))
    return log_probs[:, :-1, :]  # Drop final prediction (after all data).


class CategoricalGenerator(data_generator_base.DataGenerator):
  """I.i.d. from categorical."""

  def __init__(self, config: config_lib.CategoricalGeneratorConfig) -> None:
    chex.assert_shape(config.biases, (config.vocab_size,))
    chex.assert_trees_all_close(np.sum(config.biases), 1.0)
    self.biases = config.biases
    self.dist = distrax.OneHotCategorical(
        probs=config.biases.astype(jnp.float32)
    )
    super().__init__(config)

  def _generate(
      self,
      rng_key: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[types.Sequences, types.LogPredictions]:
    samples = self.dist.sample(
        seed=rng_key, sample_shape=(batch_size, seq_length)
    ).astype(jnp.float32)
    gt_log_probs = jnp.tile(jnp.log(self.biases), [batch_size, seq_length, 1])
    return samples, gt_log_probs

  def _solve(self, sequences: types.Sequences) -> types.LogPredictions:
    batch_size, seq_length = sequences.shape[:2]
    # "Bayes-optimal" predictions are just the ground truth, as there is no
    # latent variable to infer.
    return jnp.tile(jnp.log(self.biases), [batch_size, seq_length, 1])


class MixtureOfCategoricalsGenerator(data_generator_base.DataGenerator):
  """Mixture of categoricals."""

  def __init__(
      self, config: config_lib.MixtureOfCategoricalsGeneratorConfig
  ) -> None:
    chex.assert_rank(config.mixing_weights, 1)
    chex.assert_shape(
        config.biases, (config.mixing_weights.shape[0], config.vocab_size)
    )
    chex.assert_trees_all_close(np.sum(config.mixing_weights), 1.0)
    chex.assert_trees_all_close(np.sum(config.biases, axis=-1), 1.0)

    self._biases = config.biases
    self._mixing_weights = config.mixing_weights
    self.prior = distrax.Categorical(probs=config.mixing_weights)
    self.components_distr = distrax.OneHotCategorical(probs=self._biases)
    super().__init__(config)

  def _generate(
      self,
      rng_key: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[types.Sequences, types.LogPredictions]:
    prior_rng, data_rng = jrandom.split(rng_key, 2)
    mix_idx = self.prior.sample(seed=prior_rng, sample_shape=(batch_size))
    samples = self.components_distr.sample(
        seed=data_rng, sample_shape=(batch_size, seq_length)
    )
    samples = samples[np.arange(batch_size), :, mix_idx, :].astype(np.float32)
    log_bias_per_sequence = np.log(self._biases)[mix_idx]
    gt_log_probs = jnp.tile(
        log_bias_per_sequence[:, jnp.newaxis, :], [1, seq_length, 1]
    )
    return samples, gt_log_probs

  def _solve(self, sequences: types.Sequences) -> types.LogPredictions:
    batch_size, _, vocab_size = sequences.shape
    # Add uniform prediction before having seen any data.
    observations = np.concatenate(
        [np.zeros((batch_size, 1, vocab_size)), sequences], axis=1
    )
    # Count "heads and tails" per sequence.
    cum_sum = jnp.cumsum(observations, axis=1)
    # The likelihood for a single mixture component is (for binary vocab):
    #   p(sequence | biases[i]) = biases[i][0]^{heads} + biases[i][1]^{tails}
    # Evaluate the log likelihoods for all components independently:
    log_lhs = (
        jnp.log(self._biases)[:, None, None, :] * cum_sum[None, :, :, :]
    ).sum(-1)
    # (first dimension of result is the mixture component).

    # Get the evidence by marginalizing over the mixture components. For two
    # components we get:
    #    p(sequence) = mixing_weights[0] * p(sequence | biases[0])
    #                + mixing_weights[1] * p(sequence | biases[1])
    log_joint = log_lhs + jnp.log(self._mixing_weights)[:, None, None]
    log_evidence = jsp.special.logsumexp(log_joint, axis=0, keepdims=True)

    # Normalize to get the posterior over the mixture component (note that the
    # prior p(i) = mixing_weights[i]):
    #   p(i | sequence) = p(sequence | i) p(i) / p(sequence)
    #                   = p(sequence, i) / p(sequence)
    log_post = log_joint - log_evidence
    # Predict by weighing each mixture predictor (fixed bias) with its posterior
    # probability and marginalizing.
    #   p(next | sequence) = \sum_i p(next | i) p(i | sequence)
    log_posterior_weighted_preds = (
        jnp.log(self._biases)[:, None, None, :] + log_post[..., None]
    )
    log_probs = jsp.special.logsumexp(log_posterior_weighted_preds, axis=0)
    return log_probs[:, :-1, :]
