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

"""Base class for data generators."""

import abc

import chex
import optax

from thunnini.src import config as config_lib
from thunnini.src import types


class DataGenerator(abc.ABC):
  """Base class for data generators.

  Subclass from here and implement `_generate` and `_solve`.
  """

  def __init__(self, config: config_lib.DataGeneratorConfig):
    self.config = config

  @abc.abstractmethod
  def _generate(
      self,
      rng_key: chex.PRNGKey,
      batch_size: int,
      seq_length: int,
  ) -> tuple[types.Sequences, types.LogPredictions]:
    """Returns one-hot sequences and ground truth log probabilities."""

  @abc.abstractmethod
  def _solve(self, sequences: types.Sequences) -> types.LogPredictions:
    """Returns Bayes-optimal log probabilities for sequences."""

  def generate(
      self,
      rng_key: chex.PRNGKey | None,
      return_ground_truth_log_probs: bool = True,
  ) -> types.Sequences | tuple[types.Sequences, types.LogPredictions]:
    """Returns one-hot sequences given a random key."""
    samples, gt_log_probs = self._generate(
        rng_key, self.config.batch_size, self.config.sequence_length
    )
    # Make sure that implementation of _generate has correct shape.
    chex.assert_equal_shape([samples, gt_log_probs])
    if return_ground_truth_log_probs:
      return samples, gt_log_probs
    return samples

  def solve(
      self, sequences: types.Sequences
  ) -> tuple[types.LogPredictions, types.Losses]:
    """Returns Bayes-optimal log probabilities and losses for sequences."""
    log_probs = self._solve(sequences)
    losses = self.instant_log_loss_from_logits(log_probs, sequences)
    # Make sure that implementation of _solve has correct shape.
    chex.assert_equal_shape([log_probs, sequences])
    chex.assert_equal_shape_prefix([log_probs, losses], 2)
    return log_probs, losses

  def generate_solve_and_losses(
      self,
      rng_key: chex.PRNGKey,
  ) -> tuple[
      types.Sequences,
      types.LogPredictions,
      types.Losses,
      types.LogPredictions,
      types.Losses,
  ]:
    """Sample, compute Bayes-optimal solution and ground truth and their losses.

    Args:
      rng_key: Random seed.

    Returns:
      sequences: One hot samples.
      bayes_opt_log_probs: Log probabilities of a Bayes-optimal predictor, that
        knows the pior but not the ground truth probabilities. First prediction
        is before having seen any data.
      bayes_opt_losses: Log loss of Bayes-optimal predictor. This is the lowest
        achievable loss (on average) when only knowing the right model class
        and prior.
      gt_log_probs: ground-truth log probabilities.
      gt_losses: Log loss for ground-truth probabilities. This is the lowest
        achievable loss (on average).
    """
    sequences, gt_log_probs = self.generate(
        rng_key, return_ground_truth_log_probs=True
    )
    gt_losses = self.instant_log_loss_from_logits(gt_log_probs, sequences)
    bayes_opt_log_probs, bayes_opt_losses = self.solve(sequences)

    return (
        sequences,
        bayes_opt_log_probs,
        bayes_opt_losses,
        gt_log_probs,
        gt_losses,
    )

  def instant_log_loss_from_logits(
      self, logits: types.LogPredictions, one_hot_sequences: types.Sequences
  ) -> types.Losses:
    """Per-sample log loss for batch of sequences of one-hot samples."""
    chex.assert_equal_shape([logits, one_hot_sequences])
    return optax.safe_softmax_cross_entropy(logits, one_hot_sequences)
