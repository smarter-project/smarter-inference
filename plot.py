import argparse
import json
import pathlib
import pickle
from statistics import mean
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.axes import Axes
import sys

from admission_controller.tests.helpers.ac_perf_analyzer.ac_helper import (
    AC_Helper,
)
from admission_controller.tests.helpers.plotting import (
    plot_hit_deadline_rates,
    plot_inference_time_intervals,
    plot_results_default,
    plot_system_util,
)


def set_size(width_pt, height=None, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document width in points
    height: float
            Height of figure in inches
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 80

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if not height:
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = height

    return (fig_width_in, fig_height_in)


def plot_triton_util(ax: Axes, system_utilization):
    # Plot system cpu utilization and projected utilization
    ax.plot(
        *zip(*system_utilization["triton_cpu_util"]),
        label="CPU Actual",
    )
    ax.plot(
        *zip(*system_utilization["projected_cpu_util"]),
        label="CPU Projected",
    )
    ax.plot(
        *zip(*system_utilization["triton_memory_util"]),
        label="Mem Actual",
    )
    ax.plot(
        *zip(*system_utilization["projected_triton_memory_util"]),
        label="Mem Projected",
    )

    ax.set_title("Triton System Utilization")
    ax.set_ylabel("%")
    ax.grid()
    ax.legend(loc="center right", bbox_to_anchor=(-0.1, 0.5))
    ax.label_outer()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_yticks(range(0,101,25))


def false_positive(result):
    if (
        result["expected_deadline_misses"] == 0
        and result["total_deadline_misses"] > 0
    ):
        # Missed deadlines when not predicted, outside of miss tolerance
        for model in result["model_data"]:
            if model["hit_deadline_rate"] < args.miss_tolerance:
                return True
    return False


def false_negative(result):
    # Didn't miss deadlines when predicted
    if (
        result["expected_deadline_misses"] > 0
        and result["total_deadline_misses"] == 0
    ):
        return True
    return False


def get_fail_lists(results):
    false_positive_buckets: Dict = {x: [] for x in range(0, 100, 10)}
    false_negative_buckets: Dict = {x: [] for x in range(0, 100, 10)}
    labels = [f">{k}" for k in false_positive_buckets]

    for result in results:
        index = min(90, int(result["triton_cpu_util"] / 10) * 10)
        false_positive_buckets[index].append(int(false_positive(result)))
        false_negative_buckets[index].append(int(false_negative(result)))

    print(f"fp: {false_positive_buckets}")
    print(f"fn: {false_negative_buckets}")

    # Handle false positives
    fp_bins = {}
    for i, bucket in enumerate(false_positive_buckets):
        if false_positive_buckets[bucket]:
            fp_bins[bucket] = mean(false_positive_buckets[bucket])
        else:
            fp_bins[bucket] = 0

    fp_rate = []
    for bin in fp_bins:
        fp_rate.append(fp_bins[bin] * 100)

    # Handle false negatives
    fn_bins = {}
    for bucket in false_negative_buckets:
        if false_negative_buckets[bucket]:
            fn_bins[bucket] = mean(false_negative_buckets[bucket])
        else:
            fn_bins[bucket] = 0

    fn_rate = []
    for bin in fn_bins:
        fn_rate.append(fn_bins[bin] * 100)

    return fp_rate, fn_rate, labels


def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y) * 100)


def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


def sanitize_name(input: str):
    return "".join(input.split("-")[:-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        required=True,
        help="filenames data to use",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        required=False,
        default="example_1.png",
        help="filename to write pdf to",
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Plot with default settings",
    )
    parser.add_argument(
        "--acc-data",
        action="store_true",
        help="Plot acc results on a scatter plot",
    )
    parser.add_argument(
        "--clip-names",
        action="store_true",
        help="Clip the names on the legends",
    )
    parser.add_argument(
        "--remove-max-threads",
        action="store_true",
        help="Remove max threads",
    )
    parser.add_argument(
        "--just-under",
        action="store_true",
        help="Only plot thread counts < cores",
    )
    parser.add_argument(
        "--sanitize-names",
        action="store_true",
        help="Remove hash from model names",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=516.0,
        help="Width in points of target document",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=516.0,
        help="Width in points of target document",
    )
    parser.add_argument(
        "--miss-tolerance",
        type=float,
        default=95.0,
        help="Miss rate of client considered failure",
    )
    parser.add_argument(
        "--mark-reject",
        type=float,
        required=False,
        default=None,
        help="Mark time series plots at time",
    )

    args, unknown = parser.parse_known_args()

    if len(args.files) == 1:
        input_file = args.files[0]

    extension = pathlib.Path(args.files[0]).suffix

    if extension == ".obj":
        # Deserialize test data
        if len(args.files) > 1:
            print("Only pass 1 obj file")
            sys.exit(-1)
        with open(args.files[0], "rb") as test_results_file:
            test_data = pickle.load(test_results_file)
        system_utilization = test_data["system_utilization"]
        hit_deadline_percentages = test_data["hit_deadline_percentages"]
        ac_helpers = test_data["ac_helpers"]
        perf_stats_df = test_data["perf_stats_df"]

        if args.sanitize_names:
            hit_deadline_percentages = {sanitize_name(k): v for k,v in hit_deadline_percentages.items()}

        if args.default:
            if not (args.out_file):
                plt_name = args.files.replace(".obj", ".pdf").replace(
                    "raw_test_data", "plots"
                )
            else:
                plt_name = args.out_file

            plot_results_default(
                system_utilization=system_utilization,
                hit_deadline_percentages=hit_deadline_percentages,
                ac_helpers=ac_helpers,
                capture_duration=40,
                plt_name=plt_name,
                perf_stats_df=perf_stats_df,
            )

        # plt.style.use(['science','ieee'])
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 7
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color", sns.color_palette("hls", 15)
        )
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        fig, ax = plt.subplots(
            3, 1, figsize=set_size(args.width, fraction=1.0), sharex=True
        )
        fig.supxlabel("Time (s)")
        plot_triton_util(ax[0], system_utilization)
        plot_hit_deadline_rates(ax[1], hit_deadline_percentages)
        plot_inference_time_intervals(ax[2], ac_helpers, color_list=colors, client_ids=False)
        for a in ax:
            if args.mark_reject:
                a.axvline(x=args.mark_reject, ymin=0, color="xkcd:purple")
        plt.tight_layout()
        fig.savefig(args.out_file, format="svg", bbox_inches="tight")

    elif extension == ".json":

        results = []
        for file in args.files:
            with open(file, "r") as f:
                results.extend(json.load(f))

        if args.remove_max_threads:
            new_results = []
            for result in results:
                avail_cpus = result["avail_triton_cpus"]
                valid = True
                for model in result["model_data"]:
                    if (
                        model["threads"] >= avail_cpus
                        or model["accel_threads"] >= avail_cpus
                    ):
                        valid = False
                if valid:
                    new_results.append(result)

            results = new_results

        if args.just_under:
            new_results = []
            for result in results:
                avail_cpus = result["avail_triton_cpus"]
                total_threads = sum(
                    [
                        max(model["accel_threads"], model["threads"])
                        for model in result["model_data"]
                    ]
                )
                if total_threads < avail_cpus:
                    new_results.append(result)

            results = new_results

        print(f"Loaded {len(results)} results")

        # mispredicts_under_subscribed = []
        mispredicts_over_subscribed_high_thread = []
        mispredicts_over_subscribed_low_thread = []
        mispredicts_thread_ratio = []
        mispredicts_period = []
        # correct_preds_thread_ratio = []
        cpu_utils_projected = np.array(
            [result["projected_cpu_util"] for result in results]
        )
        cpu_utils_actual = np.array(
            [result["triton_cpu_util"] for result in results]
        )
        mem_projected = np.array(
            [result["triton_proj_mem"] for result in results]
        )
        mem_actual = np.array([result["triton_memory"] for result in results])
        print(f"MAPE cpu util: {mape(cpu_utils_actual, cpu_utils_projected)}")
        print(f"RMSE cpu util: {rmse(cpu_utils_actual, cpu_utils_projected)}")
        print(f"MAPE mem: {mape(mem_actual, mem_projected)}")
        print(f"RMSE mem: {rmse(mem_actual, mem_projected)}")
        for result in results:
            expected_deadline_misses = result["expected_deadline_misses"]
            total_deadline_misses = result["total_deadline_misses"]
            projected_cpu_util = result["projected_cpu_util"]
            hyperperiod = result["hyperperiod"]
            avail_triton_cpus = result["avail_triton_cpus"]
            total_threads = sum(
                [
                    max(model["accel_threads"], model["threads"])
                    for model in result["model_data"]
                ]
            )
            thread_to_core_ratio = total_threads / avail_triton_cpus
            triton_cpu_util = result["triton_cpu_util"]

            for model in result["model_data"]:
                if expected_deadline_misses == 0 and total_deadline_misses > 0:
                    if model["hit_deadline_rate"] < args.miss_tolerance:
                        if thread_to_core_ratio > 1:
                            mispredicts_over_subscribed_high_thread.append(
                                [triton_cpu_util, model["hit_deadline_rate"]]
                            )
                        else:
                            mispredicts_over_subscribed_low_thread.append(
                                [triton_cpu_util, model["hit_deadline_rate"]]
                            )

                        mispredicts_thread_ratio.append(
                            [thread_to_core_ratio, model["hit_deadline_rate"]]
                        )
                        mispredicts_period.append(
                            [
                                model["period"] * 1000,
                                model["hit_deadline_rate"],
                            ]
                        )

        # Handle plotting inf result data
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 7
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color", sns.color_palette("hls", 15)
        )
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # Plot scatter plot of mispredictions by cpu utilization
        # fig, ax = plt.subplots(1, 1, figsize=set_size(args.width, fraction=0.5), sharex=True)
        # if mispredicts_over_subscribed_high_thread:
        #     ax.scatter(*zip(*mispredicts_over_subscribed_high_thread), 7)
        # if mispredicts_under_subscribed:
        #     ax.scatter(*zip(*mispredicts_over_subscribed_low_thread), 7)
        # ax.set_title("Misprediction by CPU Utilization")
        # ax.set_xlabel("CPU Utilization %")
        # ax.set_ylabel("Client Deadline Hit Rate %")
        # ax.grid()
        # ax.label_outer()
        # ax.set_xlim(left=0, right=100)
        # ax.set_ylim(bottom=0)
        # fig.savefig("misspred_by_cpu_util_" + args.out_file, format='svg', bbox_inches='tight')

        fig, axs = plt.subplots(
            3, 1, figsize=set_size(args.width, 5, fraction=0.5), sharex=False
        )

        # Plot stacked bar chart with success vs failure rates:
        fp_rates, fn_rates, labels = get_fail_lists(results)
        tn_rates = [100.0 - x for x in fp_rates]
        tp_rates = [100.0 - x for x in fn_rates]
        axs[0].set_title("Prediction Accuracy by CPU Utilization")
        axs[0].set_xlabel("CPU Utilization %")
        axs[0].set_ylabel("Rate %")
        x_axis = np.arange(len(labels))
        axs[0].bar(
            x_axis - 0.2,
            fp_rates,
            0.35,
            label="Incorrect Accept",
            color=colors[0],
        )
        # ax.bar(x_axis -0.2, tn_rates, 0.35, label='Correct Accept', color=colors[4])
        axs[0].bar(
            x_axis + 0.2,
            fn_rates,
            0.35,
            label="Incorrect Reject",
            color=colors[1],
        )
        # ax.bar(x_axis +0.2, tp_rates, 0.35, label='Correct Reject', color=colors[6])
        axs[0].legend()
        axs[0].set_xticks(x_axis, labels)
        axs[0].set_ylim(bottom=0, top=100)

        if mispredicts_thread_ratio:
            axs[1].scatter(*zip(*mispredicts_thread_ratio), 7)
        axs[1].set_title("Misprediction by Thread Subscription")
        axs[1].set_xlabel("Threads / Cores")
        axs[1].set_ylabel("Client Deadline Hit Rate %")
        axs[1].grid()
        axs[1].set_xlim(left=0)
        axs[1].set_ylim(bottom=0)
        axs[1].set_ylim(bottom=0, top=100)

        # Plot scatter plot of mispredictions by period
        # fig, ax = plt.subplots(1, 1, figsize=set_size(args.width, fraction=0.5), sharex=True)
        if mispredicts_period:
            axs[2].scatter(*zip(*mispredicts_period), 7)
        axs[2].set_title("Misprediction by Client Request Period")
        axs[2].set_xlabel("Period (ms)")
        axs[2].set_ylabel("Client Deadline Hit Rate %")
        axs[2].grid()
        axs[2].set_xlim(left=0)
        axs[2].set_ylim(bottom=0, top=100)
        # fig.savefig("misspred_by_period_" + args.out_file, format='svg', bbox_inches='tight')

        plt.tight_layout()
        fig.savefig(
            "m_analy_" + args.out_file, format="svg", bbox_inches="tight"
        )
