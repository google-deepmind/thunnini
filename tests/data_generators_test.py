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

"""Tests for data generators.

To add a new data generator, add a config to the top of this file, and add this
config to the parameter list for `test_sample_from_datagen`. Also add a test to
verify that the statistics of the data generator are correct at the end of this
file.

All locations to add something when testing a new data generator are marked
with `# ------ Add your...> here -------`.
"""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np

from thunnini.src import builders
from thunnini.src import config as config_lib


# ------ Add your config here -------
CATEGORICAL_CONFIG = config_lib.CategoricalGeneratorConfig(
    batch_size=128,
    sequence_length=100,
    vocab_size=2,
    biases=np.array([0.2, 0.8]),
)

CATEGORICAL_MIXTURE_CONFIG = config_lib.MixtureOfCategoricalsGeneratorConfig(
    batch_size=128,
    sequence_length=100,
    vocab_size=2,
    mixing_weights=np.array([0.25, 0.75]),
    biases=np.array([[0.2, 0.8], [0.8, 0.2]]),
)

DIRICHLET_CONFIG = config_lib.DirichletCategoricalGeneratorConfig(
    batch_size=128,
    sequence_length=100,
    vocab_size=2,
    alphas=np.array([1, 1]),
)


class DataGeneratorsTest(parameterized.TestCase):

  # ------ Add your config here -------
  @parameterized.parameters(
      [CATEGORICAL_CONFIG, CATEGORICAL_MIXTURE_CONFIG, DIRICHLET_CONFIG]
  )
  def test_sample_from_datagen(self, config):
    """Tests that datagenerator builds and samples/solutions are deterministic."""
    datagen = builders.build_datagen(config)
    samples, bayes_opt_log_probs, bayes_opt_losses, gt_log_probs, gt_losses = (
        datagen.generate_solve_and_losses(jax.random.PRNGKey(1337))
    )
    # Draw another sample with same rng-seed and recompute solutions and losses.
    samples2, gt_log_probs2 = datagen.generate(
        jax.random.PRNGKey(1337), return_ground_truth_log_probs=True
    )
    gt_log_losses2 = datagen.instant_log_loss_from_logits(
        gt_log_probs2, samples2
    )
    bayes_opt_log_probs2, bayes_opt_losses2 = datagen.solve(samples2)
    with self.subTest('samples_and_solutions_shape'):
      chex.assert_shape(
          samples,
          (config.batch_size, config.sequence_length, config.vocab_size),
      )
      chex.assert_shape(
          bayes_opt_log_probs,
          (config.batch_size, config.sequence_length, config.vocab_size),
      )
      chex.assert_shape(
          bayes_opt_losses,
          (config.batch_size, config.sequence_length),
      )
      chex.assert_shape(
          gt_log_probs,
          (config.batch_size, config.sequence_length, config.vocab_size),
      )
      chex.assert_shape(
          gt_losses,
          (config.batch_size, config.sequence_length),
      )
    with self.subTest('datagen_deteriministic'):
      chex.assert_trees_all_equal(samples, samples2)
      chex.assert_trees_all_equal(bayes_opt_log_probs, bayes_opt_log_probs2)
      chex.assert_trees_all_equal(gt_log_probs, gt_log_probs2)
      chex.assert_trees_all_equal(bayes_opt_losses, bayes_opt_losses2)
      chex.assert_trees_all_equal(gt_losses, gt_log_losses2)

  # ------ Add your data generator here -------
  def test_categorical_generator_statistics(self):
    datagen = builders.build_datagen(CATEGORICAL_CONFIG)
    _, bayes_opt_log_probs, _, gt_log_probs, _ = (
        datagen.generate_solve_and_losses(jax.random.PRNGKey(90210))
    )
    probabilites = np.tile(
        CATEGORICAL_CONFIG.biases,
        [CATEGORICAL_CONFIG.batch_size, CATEGORICAL_CONFIG.sequence_length, 1],
    )
    # Check that probabilities match (roughly) in each step.
    with self.subTest('ground_truth_probability'):
      chex.assert_trees_all_close(np.exp(gt_log_probs), probabilites, rtol=1e-4)
    with self.subTest('bayes_optimal_probability'):
      chex.assert_trees_all_close(
          np.exp(bayes_opt_log_probs), probabilites, rtol=1e-4
      )

  def test_dirichlet_categorical_generator_statistics(self):
    datagen = builders.build_datagen(DIRICHLET_CONFIG)
    _, bayes_opt_log_probs, _, gt_log_probs, _ = (
        datagen.generate_solve_and_losses(jax.random.PRNGKey(90210))
    )
    mean_gt_probabilities = np.mean(np.exp(gt_log_probs), axis=(0, 1))
    # Compute difference between Bayes-optimal and ground-truth probabilities
    # after some "burn-in" time. Since Bayes-optimal should converge to the
    # ground-truth per sequence, this should be small.
    difference = np.abs(np.exp(bayes_opt_log_probs) - np.exp(gt_log_probs))
    index = DIRICHLET_CONFIG.sequence_length - 1
    difference = difference[:, index:, :]
    mean_difference = np.mean(difference, axis=(0, 1))
    with self.subTest('ground_truth_probability'):
      # Check that mean ground-truth probabilities are close to prior.
      chex.assert_trees_all_close(
          mean_gt_probabilities, np.array([0.5, 0.5]), rtol=1e-1
      )
    with self.subTest('bayes_optimal_probability'):
      # Check if difference is close to zero, use absolute tolerance since
      # the relative difference is very large due to comparing against zero.
      chex.assert_trees_all_close(mean_difference, np.array([0, 0]), atol=1e-1)

  def test_mixture_of_categorical_generator_statistics(self):
    datagen = builders.build_datagen(CATEGORICAL_MIXTURE_CONFIG)
    _, bayes_opt_log_probs, _, gt_log_probs, _ = (
        datagen.generate_solve_and_losses(jax.random.PRNGKey(90210))
    )
    mean_gt_probabilities = np.mean(np.exp(gt_log_probs), axis=(0, 1))
    biases = CATEGORICAL_MIXTURE_CONFIG.biases
    weights = CATEGORICAL_MIXTURE_CONFIG.mixing_weights
    mean_target_probs = weights[0] * biases[0] + weights[1] * biases[1]
    # Compute difference between Bayes-optimal and ground-truth probabilities
    # after some "burn-in" time. Since Bayes-optimal should converge to one of
    # the two ground-truth probabilities per sequence, this should be small.
    difference = np.abs(np.exp(bayes_opt_log_probs) - np.exp(gt_log_probs))
    index = CATEGORICAL_MIXTURE_CONFIG.sequence_length - 10
    difference = difference[:, index:, :]
    mean_difference = np.mean(difference, axis=(0, 1))
    with self.subTest('ground_truth_probability'):
      # Check that mean ground-truth probabilities are close to prior.
      chex.assert_trees_all_close(
          mean_gt_probabilities, mean_target_probs, rtol=5e-2
      )
    with self.subTest('bayes_optimal_probability'):
      # Check if difference is close to zero, use absolute tolerance since
      # we relative difference is very large due to comparing against zero.
      chex.assert_trees_all_close(mean_difference, np.array([0, 0]), atol=1e-1)
    with self.subTest('bayes_optimal_first_step'):
      # Check if Bayes-optimal is equal to prior in the first step.
      chex.assert_trees_all_close(
          np.exp(bayes_opt_log_probs[:, 0, :]),
          np.tile(
              mean_target_probs, [CATEGORICAL_MIXTURE_CONFIG.batch_size, 1]
          ),
          rtol=1e-4,
      )


if __name__ == '__main__':
  absltest.main()
