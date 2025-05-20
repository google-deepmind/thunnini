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

"""Helper functions for plotting results over many models / repetitions."""

import collections
from collections.abc import Callable, Iterable
import dataclasses
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def get_color_by_index(index: int, colors: list[Any] | None = None) -> Any:
  """Returns (default) color by index."""
  if colors is None:
    return f"C{index}"
  else:
    return colors[index]


@dataclasses.dataclass(kw_only=True)
class MeanAndVariabilityPlotConfig:
  """Default settings for plot_mean_and_variability_band."""

  x_vector: np.ndarray | None = None
  color: str | np.ndarray = "darkblue"
  alpha: float = 0.25
  label: str = ""  # Legend label.
  aggregate_linewidth: float = 1.5  # Aggregate line (e.g., mean).
  # How to aggregate over lines: mean or median usually
  aggregate_fn: Callable[[np.ndarray], np.ndarray] = np.median
  # How to show variability vor lines: mean +/- 2*std, or median +/- quantiles.
  variability_fn_upper: Callable[[np.ndarray], np.ndarray] = (
      lambda x: np.quantile(x, 0.75)
  )
  variability_fn_lower: Callable[[np.ndarray], np.ndarray] = (
      lambda x: np.quantile(x, 0.25)
  )
  apply_axis: int = 0  # Axis to apply aggregation function along.
  aggregate_fn_only: bool = False  # Only aggregate, or also variability band?
  fill_band: bool = True  # Fill variability band with color?
  non_fill_band_linewidth: float = 0.5  # Variability band borders.
  non_fill_band_linestyle: str = ":"  # If variability band is not filled,
  # use this linestyle.
  show_individual_lines: bool = False  # Show individual lines?
  individual_linewidth: float = 0.5  # Individual lines (if shown).
  show_gridlines: bool = False


def plot_mean_and_variability_band(
    lines: Iterable[Any],
    axis: plt.Axes,
    plot_config: MeanAndVariabilityPlotConfig = MeanAndVariabilityPlotConfig(),
) -> tuple[plt.Line2D, plt.Line2D | None]:
  """Plots a mean line with an optional variability/error band."""

  lines = np.array(lines)
  agg = np.apply_along_axis(
      plot_config.aggregate_fn, axis=plot_config.apply_axis, arr=lines
  )

  if plot_config.x_vector is None:
    x_vector = np.arange(agg.shape[0])
  else:
    x_vector = plot_config.x_vector

  agg_ln = axis.plot(
      x_vector,
      agg,
      color=plot_config.color,
      label=plot_config.label,
      linewidth=plot_config.aggregate_linewidth,
  )

  if plot_config.show_individual_lines:
    for line in lines:
      axis.plot(
          x_vector,
          line,
          color=plot_config.color,
          linewidth=plot_config.individual_linewidth,
          alpha=plot_config.alpha,
      )

  if not plot_config.aggregate_fn_only:
    var_upper = np.apply_along_axis(
        plot_config.variability_fn_upper, axis=plot_config.apply_axis, arr=lines
    )
    var_lower = np.apply_along_axis(
        plot_config.variability_fn_lower, axis=plot_config.apply_axis, arr=lines
    )

    if plot_config.fill_band:
      var_band = axis.fill_between(
          x_vector,
          var_lower,
          var_upper,
          alpha=plot_config.alpha,
          color=plot_config.color,
      )
    else:
      var_ln_lo = axis.plot(
          x_vector,
          var_lower,
          color=plot_config.color,
          alpha=plot_config.alpha,
          linewidth=plot_config.non_fill_band_linewidth,
          linestyle=plot_config.non_fill_band_linestyle,
      )
      var_ln_hi = axis.plot(
          x_vector,
          var_upper,
          color=plot_config.color,
          alpha=plot_config.alpha,
          linewidth=plot_config.non_fill_band_linewidth,
          linestyle=plot_config.non_fill_band_linestyle,
      )
      var_band = [var_ln_lo, var_ln_hi]
  else:
    var_band = None

  if plot_config.show_gridlines:
    axis.grid(True)
    axis.set_axisbelow(True)

  return agg_ln, var_band


def bar_plot_mean_and_variability_band(
    lines: Iterable[Any],
    line_index: int,
    x_location: int,
    axis: plt.Axes,
    plot_config: MeanAndVariabilityPlotConfig = MeanAndVariabilityPlotConfig(),
) -> None:
  """Plots mean bar and variability error bar at location_index."""
  lines = np.array(lines)
  agg = np.apply_along_axis(
      plot_config.aggregate_fn, axis=plot_config.apply_axis, arr=lines
  )

  var_upper = None
  var_lower = None
  if not plot_config.aggregate_fn_only:
    var_upper = np.apply_along_axis(
        plot_config.variability_fn_upper, axis=plot_config.apply_axis, arr=lines
    )
    var_lower = np.apply_along_axis(
        plot_config.variability_fn_lower, axis=plot_config.apply_axis, arr=lines
    )

  axis.bar(
      x_location,
      agg[line_index],
      color=plot_config.color,
      edgecolor="black",
      linewidth=1.5,
  )

  if not plot_config.aggregate_fn_only:
    e_up = var_upper[line_index] - agg[line_index]
    e_low = agg[line_index] - var_lower[line_index]
    axis.errorbar(
        x_location,
        y=agg[line_index],
        yerr=np.array([[e_low], [e_up]]),
        color="black",
        fmt="none",
        elinewidth=10,
        alpha=0.25,
    )

  axis.text(
      x=x_location,
      y=agg[line_index],
      s=f" {agg[line_index]:4.2f}",
      horizontalalignment="center",
      verticalalignment="bottom",
      rotation=90,
      fontsize=12,
      color=plot_config.color,
  )
  axis.margins(0.05, 0.1)  # Increase margins to avoid text outside axes.

  if plot_config.show_gridlines:
    axis.grid(True)
    axis.set_axisbelow(True)


def plot_performance_metric(
    metric_all_models: dict[str, Iterable[Any]],
    metric_name: str,
    axis: plt.Axes | None = None,
    model_exclude_list: list[str] | None = None,
    colors: Iterable[Any] | None = None,
    bar_plot: bool = False,
    bar_plot_line_index: int | None = None,
    **plot_kwargs: dict[str, Any],
) -> plt.Axes:
  """Plots a performance metric over dict of models / experiments.

  Args:
    metric_all_models: Dict of model names to an Iterable over metrics. E.g.,
      then model name could be 'LSTM' and the item could be a list of regret
      over time, with one entry in the list per fine-tuning seed. In the plot,
      this would lead to  one line for the metric for each model, where the line
      shows the median over repetitions and the variability band shows the 25th
      and 75th percentile.
    metric_name: Pretty name of the metric for y-axis label.
    axis: Axis to plot on. If None, creates a new figure and axis.
    model_exclude_list: List of model names to exclude from plotting.
    colors: List of colors to cycle through for plotting. If None, uses default
      property cycler.
    bar_plot: Whether to plot as a bar plot or line plot.
    bar_plot_line_index: Index to plot bar at, used to index into
      `metric_all_models`. Unused for line plot.
    **plot_kwargs: Keyword arguments to override MeanAndVariabilityPlotConfig.

  Returns:
    The axis that the figure was plotted on.
  """
  if axis is None:
    fig = plt.figure(figsize=(7, 4))
    axis = fig.gca()

  # Get default config and override
  plot_config = MeanAndVariabilityPlotConfig(**plot_kwargs)

  for i, (model_name, metric) in enumerate(metric_all_models.items()):
    if model_exclude_list and model_name in model_exclude_list:
      continue  # Skip models that we do not want to show.
    color = get_color_by_index(i, colors)
    plot_config.color = color
    plot_config.label = model_name
    if bar_plot:
      bar_plot_mean_and_variability_band(
          lines=metric,
          line_index=bar_plot_line_index,
          x_location=i,
          axis=axis,
          plot_config=plot_config,
      )
    else:
      plot_mean_and_variability_band(
          lines=metric, axis=axis, plot_config=plot_config
      )

  if bar_plot:
    model_names = list(metric_all_models.keys())
    axis.set_xticks(np.arange(len(model_names)))
    axis.set_xticklabels(model_names, rotation=60, ha="right")
    axis.set_ylabel(metric_name)
  else:
    axis.set_xlabel("Step $t$")
    axis.set_ylabel(metric_name)
    axis.legend()

  return axis


def postprocess_tuning_experiment_results(
    results: dict[str, Any],
    tuning_method_names: list[str],
    eval_data_source_names: list[str],
) -> tuple[
    collections.OrderedDict[str, Any],
    collections.OrderedDict[str, collections.OrderedDict[str, Iterable[Any]]],
    collections.OrderedDict[str, collections.OrderedDict[str, Iterable[Any]]],
    collections.OrderedDict[str, Iterable[Any]],
]:
  """Postprocesses results of tuning experiment for plotting compatibility.

  Results of a tuning experiment are stored in a nested dictionary, with
  results for each tuning method, each evaluation data source and each tuning
  repetition stored in separate sub-dictionaries. This function computes
  instant and cumulative regrets and stores them in a nested dict with the
  following format:
  [<eval_data_source_name>][<tuning_method_name>]List[<repetitions>].

  This allows, e.g., to pass `instant_regrets['eval_name']` to
  `plot_performance_metric` and plot instant regret for each tuning method
  (with error bands across repetitions, if there are multiple).

  Args:
    results: Dictionary of results, as returned by `run_tuning_experiment` in
      `tuning.py`.
    tuning_method_names: List of tuning methods names in resultsto process.
    eval_data_source_names: List of evaluation data source names in results to
      process.

  Returns:
    gt_losses: Dict of ground truth losses. One entry per evaluation data source
      name.
    instant_regrets: Dict of instant regrets, keyed by evaluation data source
      name
      then tuning method name, and potentially a lis of results as items (if
      there were multiple repetitions in the experiment).
    cumulative_regrets: Dict of cumulative regrets. Same structure as
      `instant_regrets`.
    tuning_loss_curves: Dict of tuning losses. One entry per tuning method name,
      with a list of tuning losses as items (one list item per repetition).
  """

  gt_losses = collections.OrderedDict()
  instant_regrets = collections.OrderedDict()
  cumulative_regrets = collections.OrderedDict()
  tuning_loss_curves = collections.OrderedDict()

  # Process Bayes-optimal results and store ground truth losses.
  for eval_name in eval_data_source_names:
    gt_losses[eval_name] = results["ground_truth"][eval_name]["losses"][0]
    instant_regrets[eval_name] = collections.OrderedDict()
    if "Bayes" in results:
      bo_instant_regret = np.mean(
          results["Bayes"][eval_name]["losses"][0] - gt_losses[eval_name],
          axis=0,
      )
      instant_regrets[eval_name]["Bayes"] = [bo_instant_regret]
    if "NoTuning" in results:
      notuning_instant_regret = np.mean(
          results["NoTuning"][eval_name]["losses"][0] - gt_losses[eval_name],
          axis=0,
      )
      instant_regrets[eval_name]["NoTuning"] = [notuning_instant_regret]

  # Process all tuning methods (across all evaluations and repetitions)
  for eval_name in eval_data_source_names:
    for tuning_name in tuning_method_names:
      if tuning_name in ["ground_truth", "Bayes", "NoTuning"]:
        continue
      if tuning_name not in instant_regrets[eval_name]:
        instant_regrets[eval_name][tuning_name] = []
        if "tuning_loss" in results[tuning_name]:
          tuning_loss_curves[tuning_name] = []
      for eval_run in results[tuning_name][eval_name]["losses"]:
        instant_regret = np.mean(eval_run - gt_losses[eval_name], axis=0)
        instant_regrets[eval_name][tuning_name].append(instant_regret)
      if "tuning_loss" in results[tuning_name]:
        for tuning_loss in results[tuning_name]["tuning_loss"]:
          tuning_loss_curves[tuning_name].append(tuning_loss)

  # Compute cumulative regrets
  for eval_name in instant_regrets.keys():
    cumulative_regrets[eval_name] = collections.OrderedDict()
    for tuning_name in instant_regrets[eval_name]:
      cumulative_regrets[eval_name][tuning_name] = []
      for instant_regret in instant_regrets[eval_name][tuning_name]:
        cumulative_regrets[eval_name][tuning_name].append(
            np.cumsum(instant_regret)
        )

  return (
      gt_losses,
      instant_regrets,
      cumulative_regrets,
      tuning_loss_curves,
  )
