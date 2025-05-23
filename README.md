# Thunnini

Thunnini Demo notebook: [![Demo notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/thunnini/blob/master/colabs/ThunniniDemo.ipynb)<br>
Full Thunnini experiment notebook: [![Full Experiment notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/thunnini/blob/master/colabs/ThunniniExperiment.ipynb)

For more details and use-cases, see Thunnini's main publication:<br>
**Understanding Prompt Tuning and In-Context Learning via Meta-Learning** (Genewein et al. 2025, [arXiv](https://arxiv.org/abs/2505.17010)).

---

Thunnini is an experimentation library to study and understand fundamental
aspects of fine-tuners for neural sequence predictors. Currently, 9 different
tuners, such as soft prompting, embedding tuning, or low-rank adaptation (LoRA),
are implemented on two neural architectures (LSTMs and transformers). Thunnini
is also the name of the zoological tribe of the tunas (17 species).

![Taxonomy of fine tunas for neural sequence predictors](https://raw.githubusercontent.com/google-deepmind/thunnini/master/assets/img/ThunniniTaxonomy.svg "Taxonomy of neural network fine tunas.")

Thunnini provides functionality to:

1.  **Pretrain** neural sequential predictors via log loss minimization over
    samples from a data generator.
2.  **Fine-tune** neural models to a target data distribution. Tune either
    weights (original weights, or additional weights, like in LoRA), or tune a
    prompt prefix (soft prompting).
3.  **Evaluate** tuned models' prediction performance on one or more evaluation
    data distributions.
4.  **Compare** different architectures and fine-tuning methods, and compare
    against baselines: the optimal oracle predictor, the several exact Bayesian
    predictors, and the untuned model.

*Simple experimentation:* Standard Thunnini experiments are very simple to
specify via a set of configurations (plain-text dicts or dataclasses) for data
generation, model architecture, training, and fine-tuning procedures, as well as
evaluation settings. Full reproducibility, given the same configuration, is
ensured.

*Batteries included:* Thunnini comes with easily configurable LSTMs and
transformers (decoder-only) and some data generators (coin-flip or dice-roll
sources, for which the exact Bayesian predictor is tractable). Thunnini also
comes with a notebook to run a full experimentation pipeline (pretraining,
fine-tuning, evaluation and comparison) by setting a few lines of configuration.

*Easily extendable:* Thunnini can be extended with more predictors and data
generators by implementing the respective interfaces and passing all tests.

The design philosophy is to provide a lightweight and nimble experimentation
library to study conceptual aspects of fine-tuning methods, and prompting
untrained networks, by having full understanding and control over the data
distributions. Connections to the theory of meta-learning and Bayesian
sequential prediction from SGD-based log loss minimization, can easily be
verified empirically. Thunnini aims at models and data that train or fine-tune
on a single GPU within minutes. LLM-scale experiments are (far) beyond the scope
of Thunnini.

## Usage

The fastest way to get started with Thunnini is via
Google's [colab](https://colab.research.google.com/), where you can directly run
notebooks in the cloud (the colabs will automatically clone
[Thunnini's git repo](https://github.com/google-deepmind/thunnini) when running on Colab). You can use
the following links to open the
[Full Experiment notebook](https://colab.research.google.com/github/google-deepmind/thunnini/blob/master/colabs/ThunniniExperiment.ipynb)
or the
[Demo notebook](https://colab.research.google.com/github/google-deepmind/thunnini/blob/master/colabs/ThunniniDemo.ipynb) on Colab.
Note that it is recommended that you use a GPU runtime, both locally and
on Colab. If no GPU runtime is available small experiments can be run on a CPU
by adjusting experiment settings (shorter sequences, smaller models, fewer
training/tuning steps, less repetitions).

If you want to use thunnini locally, e.g., because you want to modify or extend
the code or run python scripts instead of colab notebooks, use the instructions
in the next section. See also [how to run Thunnini's tests](#running-tests).

## Local installation and usage

Clone the source code into a local directory:

```bash
git clone https://github.com/google-deepmind/thunnini.git
cd thunnini
```

This repository requires Python 3.11. `pip install -r requirements.txt` will
install all required dependencies. This is best done inside a virtual Python
environment. To that end, install [virtualenv](https://virtualenv.pypa.io/):

```bash
sudo apt-get install virtualenv python3-venv
```

Then, create and activate the virtualenv:

```bash
virtualenv thunnini_env
source thunnini_env/bin/activate
```

Alternatively you can also use [conda](https://www.anaconda.com/) to manage
virtual environments. See the conda documentation for instructions.

Inside your virtual environment, use `pip` to install all required dependencies:

```bash
pip install -r requests
pip install -r requirements.txt
```

**Running notebooks locally:** To get started with Thunnini locally, start a
local Jupyter notebook server.

> If you followed the installation instructions above, make sure to first
> activate your virtual environment and set the `PYTHONPATH`. Change dir to the
> local clone of the `thunnini` folder and run:
>
> ```bash
> source thunnini_env/bin/activate
> export PYTHONPATH=$(pwd)/..
> ```

Start a local Jupyter notebook with:
```bash
jupyter notebook
```

This will give you a (local) URL in the terminal with an authentication token
which you need to click on or copy and paste into your browser. From there,
navigate to `thunnini/colabs` and open one of the two notebooks. Alternatively,
the local notebook server can be set as a local runtime for Colab. See
[local Colab runtimes](https://research.google.com/colaboratory/local-runtimes.html)
for instructions, including how to access Colab Docker runtime images with GPU
support.

To run easily configurable Thunnini experiments (from pretrainig via fine-tuning
to evaluation and comparison of methods) use `colabs/ThunniniExperiment.ipynb`.
The notebook allows to easily change standard configuration settings and make
small extensions. It also demonstrates Thunnini's user level functions. For more
sophisticated modifications, diving deeper into Thunnini is necessary.

If you want to get a very quick overview on how to train, fine-tune, and
evaluate a single model, use `colabs/ThunniniDemo.ipynb`; it avoids the boiler
plate code that `ThunniniExperiment.ipynb` needs to surface all configuration
options and collect and compare results across many fine-tuning methods.

*After* you have run all experiments, etc., leave your virtual environment with:

```bash
deactivate
```

## Fine-tuning methods

The following fine-tuners are implemented:

*   **Prefix tuning (4 methods):** Fine-tune a (fixed length) prefix of tokens,
    or token-embeddings, by minimizing log loss on samples from the fine-tuning
    data generator. Model weights remain frozen.
    *   *Simplex prefix:* The prompt prefix is constrained to lie in the simplex
        spanned by the D-dimensional space of one-hot tokens, i.e., the simplex
        prefix is a sequence of D-dimensional real vectors whose components sum
        to one.
    *   *Real prefix:* Same as SimplexPF but without the simplex constraint,
        i.e., a sequence of D-dimensional real vectors.
    *   *Soft prefix:* Direct tuning of the embeddings that would result from a
        prompt prefix of a certain length ([Soft Prompting](https://arxiv.org/abs/2104.08691)).
    *   *Hard prefix:* Use exhaustive search to find the best prefix sequence of
        hard (one-hot) tokens. Intractable for long prefixes and large token
        alphabets.
*   **Weight-tuning (4 methods):** Tune all or some of the model's
    weights.
    *   *Full weights:* fine-tune all weights.
    *   *Embedding:* fine-tune only weights of initial embedding layer.
    *   *Unembedding:* fine-tune only weights of final unembedding layer.
    *   *Un+Embedding:* fine-tune initial and final layer.
*   **Additional-weights tuning (1 method):** Introduce additional tunable model
    parameters, like task specific heads or adapters, and keep original model
    weights frozen.
    *   *LoRA:* [Low-rank Adaptation](https://arxiv.org/abs/2106.09685)
        introduces low-rank additive weight matrices to linear layers in
        transformer blocks (therefore it is only supported for transformers).

To ensure compatibility with all fine-tuning methods and other functionality
provided by Thunnini (such as embedding-prefixed forward passes), the
`Predictor` class wraps a `PredictorTorso` between an embedding and unembedding
layer (among other things). Currently a `LSTMPredictorTorso` and a
`TransformerPredictorTorso` are implemented, and it is highly recommended to add
new neural architectures as torsos to make them instantly compatible with all of
Thunnini's functionality.

## Data generators

Thunnini provides a number of standard data generators. Data generators allow to
draw samples and return the corresponding "ground-truth" generating
probabilities (for the oracle predictor baseline). Generators also compute their
respective Bayesian predictors (for the Bayes optimal baseline). Currently
implemented data generators are:

*   *Categorical:* Fixed categorical distribution. For 2 dimensions this is a
    Bernoulli variable, a.k.a. a coin with fixed bias.
*   *Mixture of Categoricals:* Mixture of several Categorical distributions with
    particular mixing proportions. For 2 dimensions this is a mixture over coins
    with different bias.
*   *Dirichlet-Categorical:* Distribution over Categoricals with a Dirichlet
    prior. For 2 dimensions this is Beta-Binomial, i.e., an infinite mixture
    over coins with different bias, where the probability of each bias is given
    by the Beta distribution.

## Contents

Short description of the repository contents. Files marked with a `+` are the
entry points to extend Thunnini with additional data generators or neural
predictors (torsos).

```
.
|
├── assets
|   └── img                             - Images for Readme.md
|
├── colabs                              - Main entry point to run standard experiments
|   ├── ThunniniDemo.ipynb              - Demo notebook showcasing Thunnini features
|   └── ThunniniExperiment.ipynb        - Easily configurable full experiment pipeline
|
├── src
|   ├── builders.py                     - Factories to build instances from configs
|   ├── config.py                       - Configurations (dataclasses)
|   ├── data_generator_base.py          - Base class for all data generators
|   ├── data_generators.py              + Implementations of data generators (add yours here)
|   ├── evaluation.py                   - Evaluation of trained / tuned models
|   ├── plot_utils.py                   - Hepler functions for plotting in notebooks
|   ├── predictor.py                    - Sequential predictor (only modify / extend for advanced use cases)
|   ├── predictor_torsos.py             + Implementations of neural predictor torsos (add yours here)
|   ├── predictor_tuning_functions.py   - Implementation of various tuning methods
|   ├── training.py                     - Pretraining loop
|   ├── transformer_utils.py            - Helper functions for transformers
|   ├── tuning.py                       - Fine-tuning loop (for all tuning methods)
|   └── types.py                        - Type definitions and literals
|
├── tests                               - Tests, use these to ensure compatibility for new torsos and data generators
|
├── CIATATION.cff                       - Citation info for Thunnini
├── Contributing.md                     - Contribution info.
├── LICENSE                             - Thunnini's license.
├── README.md                           - This file
└── requirements.txt                    - Dependencies
```

### Extending Thunnini

Implement additional neural predictors as torsos by inheriting from
`PredictorTorsos`. Use the current torsos in `src/predictor_torsos.py` as
examples. Make sure to add a config dataclass for the new torso, add the torso
and config in `builders.py`, to the `TorsoTypes` in `src/types.py`, and to the
predictor related tests.

> Thunnini interacts with these torsos via the `Predictor` class. Unless
> necessary do not modify the latter, since this may easily require large
> changes across the codebase, whereas adding torsos requires few changes
> outside the torso implementation.

New data generators can be added following the examples in
`src/data_generators.py`. As with predictor torsos, make sure to add a config
file, add the new data generator and config file to `builders.py`, to
`DataGenType` in `src/types.py`, and to the data generator tests.

> Thunnini requires that data generators return ground truth probabilities for
> each symbol, and implement a Bayesian predictor that takes sequences and
> yields Bayes-optimal predictions. These requirements can be lifted, but it
> will require a number of changes to the internals of the codebase.

Extensions beyond these, such as adding RL-based tuning, or going from
prediction error (log loss) minimization to return maximization in interactive
decision-making tasks, will require fairly significant modifications to
Thunnini.

### Running Tests

Thunnini comes with a set of tests that are helpful to ensure that
extensions like new data generators, or new predictors are fully compatible with
all of Thunnini's functionality. All tests are under `/tests/`, and use
[abseil](https://abseil.io/docs/python/guides/testing). To run tests, first
activate your virtual environment and set the python path. Assuming you have
followed the [installation instructions](#local-installation-and-usage), run
the following in the local clone of the `thunnini` folder:

```bash
source thunnini_env/bin/activate
export PYTHONPATH=$(pwd)/..
```

Then run all tests with:

```bash
python tests/config_test.py
python tests/data_generators_test.py
python tests/evaluation_test.py
python tests/predictor_test.py
python tests/training_test.py
python tests/tuning_test.py
```

Make sure to check the test summary per test (6 in case of the command above).

> While developing a specific feature it makes sense to only run the affected
> tests, and only run the full suite of tests at the end of development.

## Citing Thunnini

Cite Thunnini's main publication as:

```
@misc{genewein2025understanding,
      title={Understanding Prompt Tuning and In-Context Learning via Meta-Learning},
      author={Genewein, Tim and Li, Kevin Wenliang and Ruoss, Anian, and Grau-Moya, Jordi, and Orseau, Laurent, and Hutter, Marcus},
      year={2025},
      eprint={2505.17010},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17010},
}
```

If you use Thunnini in research papers, other publications, or software derived
from it, cite as:

```
@software{genewein2025thunnini,
  author = {Genewein, Tim and Li, Kevin Wenliang and Ruoss, Anian},
  month = {05},
  title = {Thunnini},
  url = {https://github.com/google-deepmind/thunnini.git},
  year = {2025}
}
```

## License and disclaimer

Copyright 2025 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
