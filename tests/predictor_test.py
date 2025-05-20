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

import copy
import typing

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import optax

from thunnini.src import builders
from thunnini.src import config as config_lib
from thunnini.src import predictor_tuning_functions
from thunnini.src import types


_PREDICTOR_CONFIG = config_lib.PredictorConfig(
    token_dimensionality=2,
    embedding_dimensionality=8,
)

_LINEAR_TORSO_CONFIG = config_lib.LinearTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
    return_hidden_states=False,
)

_TRANSFORMER_TORSO_CONFIG = config_lib.TransformerTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
    num_attention_heads=2,
    positional_encoding='SinCos',
    return_hidden_states=True,
    use_bias=False,
    widening_factor=4,
    normalize_qk=True,
    use_lora=False,
    reduced_rank=4,
)

_LORA_TRANSFORMER_TORSO_CONFIG = config_lib.TransformerTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
    num_attention_heads=2,
    positional_encoding='SinCos',
    return_hidden_states=True,
    use_bias=False,
    widening_factor=4,
    normalize_qk=True,
    use_lora=True,
    reduced_rank=4,
)

_LSTM_TORSO_CONFIG = config_lib.LSTMTorsoConfig(
    is_trainable=True,
    hidden_sizes=[16, 8],
    return_hidden_states=True,
)


class PredictorTest(parameterized.TestCase):

  @parameterized.parameters([
      _LINEAR_TORSO_CONFIG,
      _LSTM_TORSO_CONFIG,
      _TRANSFORMER_TORSO_CONFIG,
      _LORA_TRANSFORMER_TORSO_CONFIG,
  ])
  def test_predictor(self, torso_config: config_lib.PredictorTorsoConfig):
    batch_size = 64
    sequence_length = 100
    vocab_size = _PREDICTOR_CONFIG.token_dimensionality
    prompt_length = 5
    sequences = jnp.zeros((batch_size, sequence_length, vocab_size))
    sequences = sequences.at[:, :, 0].set(1)  # Make "one-hot" inputs.
    prefix_prompt = jnp.zeros((prompt_length, vocab_size))
    prefix_prompt = prefix_prompt.at[:, 1].set(1)  # Make "one-hot" prompts.
    prefix_embeddings = jnp.zeros(
        (prompt_length, _PREDICTOR_CONFIG.embedding_dimensionality)
    )
    prefix_embeddings = prefix_embeddings.at[:, 0].set(1)

    predictor = builders.build_predictor(_PREDICTOR_CONFIG, torso_config)
    init_rng = jax.random.PRNGKey(1337)

    # ----- Test forward pass functions -----
    with self.subTest('call_predictor'):
      params = predictor.init(rngs=init_rng, sequences=sequences)
      log_preds, states, prefix_log_preds, prefix_states = predictor.apply(
          params, sequences
      )
      chex.assert_equal_shape([log_preds, sequences])
      if states is not None:
        chex.assert_tree_shape_prefix(states, (batch_size, sequence_length))
      self.assertIsNone(prefix_log_preds)
      self.assertIsNone(prefix_states)
    with self.subTest('call_predictor_with_real_prefix'):
      params = predictor.init(
          rngs=init_rng,
          sequences=sequences,
          prefix_type='prepend',
          prefix=prefix_prompt,
      )
      log_preds, states, pf_log_preds, pf_states = predictor.apply(
          params,
          sequences,
          prefix_type='prepend',
          prefix=prefix_prompt,
      )
      chex.assert_equal_shape([log_preds, sequences])
      if states is not None:
        chex.assert_tree_shape_prefix(states, (batch_size, sequence_length))
      chex.assert_shape(pf_log_preds, (batch_size, prompt_length, vocab_size))
      if pf_states is not None:
        chex.assert_tree_shape_prefix(pf_states, (batch_size, prompt_length))
    with self.subTest('call_predictor_with_simplex_prefix'):
      params = predictor.init(
          rngs=init_rng,
          sequences=sequences,
          prefix_type='simplex',
          prefix=prefix_prompt,
      )
      log_preds, states, pf_log_preds, pf_states = predictor.apply(
          params,
          sequences,
          prefix_type='simplex',
          prefix=prefix_prompt,
      )
      chex.assert_equal_shape([log_preds, sequences])
      if states is not None:
        chex.assert_tree_shape_prefix(states, (batch_size, sequence_length))
      chex.assert_shape(pf_log_preds, (batch_size, prompt_length, vocab_size))
      if pf_states is not None:
        chex.assert_tree_shape_prefix(pf_states, (batch_size, prompt_length))
    with self.subTest('call_predictor_with_embedding_prefix'):
      params = predictor.init(
          rngs=init_rng,
          sequences=sequences,
          prefix_type='embedding',
          prefix=prefix_embeddings,
      )
      log_preds, states, pf_log_preds, pf_states = predictor.apply(
          params,
          sequences,
          prefix_type='embedding',
          prefix=prefix_embeddings,
      )
      chex.assert_equal_shape([log_preds, sequences])
      if states is not None:
        chex.assert_tree_shape_prefix(states, (batch_size, sequence_length))
      chex.assert_shape(pf_log_preds, (batch_size, prompt_length, vocab_size))
      if pf_states is not None:
        chex.assert_tree_shape_prefix(pf_states, (batch_size, prompt_length))

    # ----- Test update functions for all tuning methods -----
    for tuning_method in typing.get_args(types.TuningMethodType):
      if tuning_method == 'lora_finetune':
        if not isinstance(torso_config, config_lib.TransformerTorsoConfig):
          continue
        if not torso_config.use_lora:
          continue

      opt = optax.adam(learning_rate=1e-3)

      # Keep original prefix and prefix embeddings to later compare against.
      new_prefix = copy.deepcopy(prefix_prompt)
      new_prefix_embeddings = copy.deepcopy(prefix_embeddings)

      if tuning_method in [
          'full_parameters',
          'lora_finetune',
          'embedding',
          'unembedding',
          'embedding_unembedding',
      ]:
        params = predictor.init(rngs=init_rng, sequences=sequences)
        new_params = copy.deepcopy(params)
        new_prefix = None
        pf_type = 'none'
        opt_state = opt.init(params=new_params)
      elif tuning_method == 'prefix_real':
        pf_type = 'prepend'
        params = predictor.init(
            rngs=init_rng,
            sequences=sequences,
            prefix_type='prepend',
            prefix=new_prefix,
        )
        new_params = copy.deepcopy(params)
        opt_state = opt.init(params=new_prefix)
      elif tuning_method == 'prefix_simplex':
        pf_type = 'simplex'
        params = predictor.init(
            rngs=init_rng,
            sequences=sequences,
            prefix_type='simplex',
            prefix=new_prefix,
        )
        new_params = copy.deepcopy(params)
        opt_state = opt.init(params=new_prefix)
      elif tuning_method == 'prefix_soft':
        pf_type = 'embedding'
        new_prefix = new_prefix_embeddings
        params = predictor.init(
            rngs=init_rng,
            sequences=sequences,
            prefix_type='embedding',
            prefix=new_prefix_embeddings,
        )
        new_params = copy.deepcopy(params)
        opt_state = opt.init(params=new_prefix_embeddings)
      else:
        raise ValueError(f'Test for {tuning_method} tuning not implemented.')

      grad_fn = predictor_tuning_functions.make_grad_fn(
          predictor=predictor, tuning_method=tuning_method
      )

      # Run two updates (lora out params will not get nonzero gradients in the
      # first update, since lora in is initialized with zeros).
      for _ in range(2):
        new_params, new_prefix, opt_state, _, _ = (
            predictor_tuning_functions.update_parameters(
                params=new_params,
                opt_state=opt_state,
                sequences=sequences,
                grad_fn=grad_fn,
                optimizer=opt,
                prefix_type=pf_type,
                prefix=new_prefix,
                tuning_method=tuning_method,
            )
        )

      # Helper function to check that parameters have been updated.
      def check_updated(new, original, tuning_method):
        has_diff = jax.tree_util.tree_map(
            lambda a, b: not jax.numpy.allclose(a, b, rtol=1e-06),
            new,
            original,
        )
        # Get path of leaves that do not have a diff.
        has_diff_flattened = jax.tree_util.tree_leaves_with_path(has_diff)
        leaves_with_diff = []
        leaves_with_no_diff = []
        for path, diff in has_diff_flattened:
          path = '.'.join(map(str, path))
          if diff:
            leaves_with_diff.append(path)
          else:
            leaves_with_no_diff.append(path)

        if tuning_method == 'lora_finetune':
          # If we are lora finetuning then all leaves with diff should be lora
          # parameters, and all leaves with no diff should be non-lora
          # parameters.
          self.assertTrue(
              all('LoRA' in p for p in leaves_with_diff),
              'Non-lora params got updated during lora finetuning. All'
              f' params with update: {leaves_with_diff}',
          )
          self.assertFalse(
              any('LoRA' in p for p in leaves_with_no_diff),
              'Some lora params did not get updated during lora finetuning.'
              f' All params with no update: {leaves_with_no_diff}',
          )
        else:
          # If we are not lora finetuning, then all leaves with diff should be
          # non-lora parameters, and all leaves with no diff should be lora
          # parameters.
          self.assertFalse(
              any('LoRA' in p for p in leaves_with_diff),
              f'Some lora params got updated during "{tuning_method}" tuning.'
              f' All params with update: {leaves_with_diff}',
          )
          self.assertTrue(
              all('LoRA' in p for p in leaves_with_no_diff),
              'Non-lora params did not get updated during'
              f' "{tuning_method}" tuning. All params with no update:'
              f' {leaves_with_no_diff}',
          )

      with self.subTest(f'update_parameters_{tuning_method}'):
        match tuning_method:
          case 'prefix_real' | 'prefix_simplex':
            # Check gradient shape was correct.
            chex.assert_trees_all_equal_shapes(new_prefix, prefix_prompt)
            if torso_config.torso_type != 'Linear':  # Prefix-gradients are zero
              # for linear torso.
              check_updated(new_prefix, prefix_prompt, tuning_method)
            # Check parameters not changed.
            chex.assert_trees_all_equal(new_params, params)
          case 'prefix_soft':
            # Check gradient shape was correct.
            chex.assert_trees_all_equal_shapes(new_prefix, prefix_embeddings)
            if torso_config.torso_type != 'Linear':  # Prefix-gradients are zero
              # for linear torso.
              check_updated(new_prefix, prefix_embeddings, tuning_method)
            # Check parameters not changed.
            chex.assert_trees_all_equal(new_params, params)
          case 'full_parameters':
            # Check gradient shape was correct.
            chex.assert_trees_all_equal_shapes(new_params, params)
            check_updated(new_params, params, tuning_method)
          case 'lora_finetune':
            chex.assert_trees_all_equal_shapes(new_params, params)
            check_updated(new_params, params, tuning_method)
          case 'embedding' | 'unembedding' | 'embedding_unembedding':
            un_embedding_found = False
            for k in params['params']:
              if (
                  (tuning_method == 'embedding' and k == 'embedding')
                  or (tuning_method == 'unembedding' and k == 'unembedding')
                  or (
                      tuning_method == 'embedding_unembedding'
                      and k in ['embedding', 'unembedding']
                  )
              ):
                un_embedding_found = True
                # Check embedding/unembedding shape correct.
                chex.assert_trees_all_equal_shapes(
                    new_params['params'][k], params['params'][k]
                )
                check_updated(
                    new_params['params'][k], params['params'][k], tuning_method
                )
              else:
                # Check other parameters not changed.
                chex.assert_trees_all_equal(
                    new_params['params'][k], params['params'][k]
                )
            self.assertTrue(un_embedding_found)


if __name__ == '__main__':
  absltest.main()
