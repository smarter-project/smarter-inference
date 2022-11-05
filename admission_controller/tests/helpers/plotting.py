# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Generator, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from admission_controller.tests.helpers.ac_perf_analyzer.ac_helper import (
    AC_Helper,
)
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame

colors = sns.color_palette("hls", 15)


def get_axis(axes: List[Axes]) -> Generator[Axes, None, None]:
    for i in range(len(axes)):
        yield axes[i]


def get_from_gs(gs: gridspec.GridSpec, fig) -> Generator[Axes, None, None]:
    for i in range(gs.nrows):
        yield fig.add_subplot(gs[i])


def plot_results_default(
    system_utilization: Dict[str, List],
    hit_deadline_percentages: Dict[str, List],
    ac_helpers: List[AC_Helper],
    capture_duration: int,
    plt_name: str,
    perf_stats_df: DataFrame = DataFrame(),
):
    # Get starting timestamp
    ac_helper_start_time_stamps = [
        ac_helper.start_time_stamp for ac_helper in ac_helpers
    ]

    # Generate all of our axes
    base_plots = (
        3
        + len(ac_helpers)
        + (
            perf_stats_df.groupby("event").ngroups
            if not (perf_stats_df.empty)
            else 0
        )
    )
    total_plots = base_plots + len(ac_helpers)
    logging.info(f"Base plots: {base_plots}")
    fig = plt.figure(
        constrained_layout=False,
        figsize=(3 * total_plots, min(capture_duration, 40)),
    )
    subfigs = fig.subfigures(2, 1, height_ratios=[3, 1])

    # All time series plots will share x axis and be clipped for "dead zones"
    axs = subfigs[0].subplots(
        base_plots,
        sharex=True,
    )

    axs_gen = get_axis(axs)

    plot_hit_deadline_rates(next(axs_gen), hit_deadline_percentages)

    plot_system_util(next(axs_gen), system_utilization)

    for ac_helper in ac_helpers:
        plot_server_statistics(next(axs_gen), ac_helper)

    if not (perf_stats_df.empty):
        for event, frame in perf_stats_df.groupby("event"):
            plot_event_perf_stats(
                next(axs_gen),
                frame,
                event,
                [ac_helper.model_name for ac_helper in ac_helpers],
            )

    plot_inference_time_intervals(next(axs_gen), ac_helpers)

    # All time series plots will share x axis and be clipped for "dead zones"
    axs = subfigs[1].subplots(
        len(ac_helpers),
        sharex=True,
    )

    if len(ac_helpers) > 1:
        axs_gen = get_axis(axs)
    else:
        axs_gen = get_axis([axs])

    for i, ac_helper in enumerate(ac_helpers):
        plot_inference_distributions(
            next(axs_gen), ac_helper, i, ac_helper_start_time_stamps
        )

    plt.savefig(plt_name, bbox_inches="tight")


def plot_hit_deadline_rates(ax: Axes, hit_deadline_percentages):
    # Plot hit deadline rates for each model
    for key in hit_deadline_percentages:
        ax.plot(*zip(*hit_deadline_percentages[key]), label=key)
        ax.set_label(key)
        ax.set_title("Hit Deadline Percentage Rate")
        ax.set_ylabel("%")
        ax.grid()
        ax.legend(loc="center right", bbox_to_anchor=(-0.1, 0.5))
        ax.label_outer()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=105)
        ax.set_yticks(range(0, 101, 25))


def plot_system_util(ax: Axes, system_utilization):
    # Plot system cpu utilization and projected utilization
    ax.plot(
        *zip(*system_utilization["system_wide_cpu_util"]),
        label="Total CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["system_cpu_util"]),
        label="System CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["user_cpu_util"]),
        label="User CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["iowait_cpu_util"]),
        label="IO Wait CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["irq_cpu_util"]),
        label="IRQ CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["softirq_cpu_util"]),
        label="Soft IRQ CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["triton_cpu_util"]),
        label="Triton CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["projected_cpu_util"]),
        label="Triton CPU Projected",
    )
    ax.plot(
        *zip(*system_utilization["triton_memory_util"]),
        label="Triton Mem Actual",
    )
    ax.plot(
        *zip(*system_utilization["projected_triton_memory_util"]),
        label="Triton Mem Projected",
    )

    ax.set_title("System Utilization")
    ax.set_ylabel("%")
    ax.grid()
    ax.legend(loc="center right", bbox_to_anchor=(-0.1, 0.5))
    ax.label_outer()
    ax.set(xlabel="Time (s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def plot_server_statistics(ax: Axes, ac_helper: AC_Helper):
    df = ac_helper.time_series_tritonserver_data
    df.plot.area(x="Time", ax=ax, stacked=True)
    ax.set_title(f"{ac_helper.model_name} Latency Components")
    ax.set_ylabel("Time (ms)")
    ax.grid()
    ax.label_outer()
    ax.set(xlabel="Time (s)")
    ax.set_xlim(left=0)


def plot_triton_threads(ax: Axes, system_utilization):
    # Plot system resident memory
    ax.plot(
        *zip(*system_utilization["triton_threads"]),
        label="Threads",
    )
    ax.set_title("Triton Thread Count")
    ax.set_ylabel("Num Threads")
    ax.grid()
    ax.legend(loc="center right", bbox_to_anchor=(-0.1, 0.5))
    ax.label_outer()
    ax.set(xlabel="Time (s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def plot_event_perf_stats(
    ax: Axes, df: DataFrame, event: str, model_names: List[str]
):
    ylabel = None
    model_names = []
    plot_handles = []
    i = 0
    for model_name, model_df in df.groupby("group_name", sort=False):
        model_names.append(model_name)
        model_plot_handles = []
        for tid, tid_df in model_df.groupby("tid"):
            try:
                # Handle special cases
                if event == "instructions":
                    (p,) = ax.plot(
                        tid_df["elapsed"],
                        tid_df["additional"],
                        color=colors[i],
                    )
                    ylabel = "IPC"
                elif event == "cpu-clock" or event == "task-clock":
                    (p,) = ax.plot(
                        tid_df["elapsed"],
                        tid_df["additional"],
                        color=colors[i],
                    )
                    ylabel = "CPU Utilized"
                else:
                    (p,) = ax.plot(
                        tid_df["elapsed"], tid_df["counts"], color=colors[i]
                    )
                model_plot_handles.append(p)

            except Exception as e:
                # This means the data to plot was not recorded properly so just skip it
                logging.error(
                    f"Failed to plot: {model_name}_{event} with error: {e}"
                )
                logging.error(model_df)
                pass

        plot_handles.append(tuple(model_plot_handles))
        i += 1

    if not ylabel:
        try:
            ylabel = model_df["units"][0]
            if type(ylabel) != str:
                ylabel = "count"
        except KeyError:
            ylabel = "count"
    ax.grid()
    ax.set_title(event)
    ax.label_outer()
    ax.legend(
        plot_handles,
        model_names,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    ax.set(xlabel="Time (s)")
    ax.set(ylabel=ylabel)
    ax.set_xlim(left=0)


def plot_inference_distributions(
    ax: Axes,
    ac_helper: AC_Helper,
    ac_helper_index: int,
    time_stamps: List[float],
):
    hist_colors = colors[0 : len(time_stamps)]

    # Plot histogram of latencies and line representing the period
    ax.set_title(
        f"Latency Distribution {ac_helper.model_name}-{ac_helper.client_id}"
    )
    ax.axvline(x=ac_helper.period, ymin=0, color="xkcd:red")
    try:
        ax.axvline(
            x=ac_helper.projected_p99_latency / 1e3,
            ymin=0,
            color="xkcd:purple",
        )
    except AttributeError:
        pass
    ax.axvline(x=ac_helper.p99_latency, ymin=0, color="xkcd:orange")
    ax.set_xlabel("Inference Latency (s)")
    for j in range(ac_helper_index, len(time_stamps)):
        try:
            stop_time_stamp = time_stamps[j + 1]
        except IndexError:
            stop_time_stamp = None

        latencies = ac_helper.inference_latencies_from_timestamps(
            time_stamps[j], stop_time_stamp
        )

        ax.hist(
            latencies,
            bins="auto",
            color=hist_colors[j],
        )


def plot_inference_time_intervals(
    ax: Axes, ac_helpers: List[AC_Helper], color_list=None, client_ids=True
):
    ax.set_title(f"Inference Time Intervals")

    if not color_list:
        color_list = colors
    labels = [
        f"{ac_helper.model_name}-{ac_helper.client_id}"
        if client_ids
        else ac_helper.model_name
        for ac_helper in ac_helpers
    ]

    # Plot inference latencies as a stacked bar chart horizontally
    max_len = max(
        [len(ac_helper.inference_latencies) for ac_helper in ac_helpers]
    )
    for i in range(max_len):
        interval_widths = [
            ac_helper.inference_latencies[i]
            if i < len(ac_helper.inference_latencies)
            else 0
            for ac_helper in ac_helpers
        ]
        interval_starts = [
            ac_helper.inference_start_times[i]
            if i < len(ac_helper.inference_start_times)
            else 0
            for ac_helper in ac_helpers
        ]
        ax.barh(
            labels,
            width=interval_widths,
            left=interval_starts,
            align="center",
            color=color_list,
        )

    # Plot missed inference latencies as a stacked bar chart horizontally over the top
    max_len_miss = max(
        [len(ac_helper.miss_inference_latencies) for ac_helper in ac_helpers]
    )
    for i in range(max_len_miss):
        miss_interval_widths = [
            ac_helper.miss_inference_latencies[i]
            if i < len(ac_helper.miss_inference_latencies)
            else 0
            for ac_helper in ac_helpers
        ]
        miss_interval_starts = [
            ac_helper.miss_inference_start_times[i]
            if i < len(ac_helper.miss_inference_start_times)
            else 0
            for ac_helper in ac_helpers
        ]
        ax.barh(
            labels,
            width=miss_interval_widths,
            left=miss_interval_starts,
            align="center",
            color="xkcd:red",
        )
    ax.grid(axis="x")
    ax.invert_yaxis()  # labels read top-to-bottom
