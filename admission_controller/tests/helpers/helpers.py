import logging
import multiprocessing
import pickle
from datetime import datetime
from itertools import chain, combinations
from time import sleep
from timeit import default_timer as timer
from typing import Dict, List

import psutil
import requests  # type: ignore
from admission_controller.tests.helpers.ac_perf_analyzer.ac_helper import (
    AC_Helper,
)
from pandas import DataFrame
from tabulate import tabulate


def normalize_timeseries_data(
    system_utilization: Dict[str, List],
    hit_deadline_percentages: Dict[str, List],
    ac_helpers: List[AC_Helper],
    perf_stats_df: DataFrame = DataFrame(),
):
    """
    Take in filled result data from inference test and normalize it based
    on the mininum recorded time stamp from the test

    Args:
        system_utilization (Dict[str, List]): Holds system util data perf metric
        hit_deadline_percentages (Dict[str, List]): holds deadline percentage hit rates per model
        ac_helpers (List[AC_Helper]): List of ac_helpers used during test
        perf_stats_df (DataFrame, optional): Holds perf data in df if used during test. Defaults to DataFrame().
    """
    # Get starting timestamps and min time stamp
    ac_helper_start_time_stamps = [
        ac_helper.start_time_stamp for ac_helper in ac_helpers
    ]
    min_start_time_stamp_ac_helper = min(ac_helper_start_time_stamps)

    # Trim and normalize the time series data based on start_time_stamp
    for metric_list in system_utilization.values():
        for x in metric_list:
            if x[0] > min_start_time_stamp_ac_helper:
                x[0] -= min_start_time_stamp_ac_helper
            else:
                del x

    for hit_list in hit_deadline_percentages.values():
        for x in hit_list:
            if x[0] > min_start_time_stamp_ac_helper:
                x[0] -= min_start_time_stamp_ac_helper
            else:
                del x

    for ac_helper in ac_helpers:
        # Normalize the tritonserver data
        ac_helper.time_series_tritonserver_data["Time"] = (
            ac_helper.time_series_tritonserver_data["Time"]
            - min_start_time_stamp_ac_helper
        )

        # Normalize the inference start times
        ac_helper.start_time_stamp -= min_start_time_stamp_ac_helper
        for i in range(len(ac_helper.inference_start_times)):
            ac_helper.inference_start_times[
                i
            ] -= min_start_time_stamp_ac_helper

        # Normalize the miss inference start times
        for i in range(len(ac_helper.miss_inference_start_times)):
            ac_helper.miss_inference_start_times[
                i
            ] -= min_start_time_stamp_ac_helper

    if not (perf_stats_df.empty):
        # Normalize start time stamps to min start time
        perf_stats_df["elapsed"] = (
            perf_stats_df["elapsed"] - min_start_time_stamp_ac_helper
        )


def powerset(iterable) -> List:
    s = list(iterable)  # allows duplicate elements
    return list(
        chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    )


def save_run_results(
    file_name: str,
    system_utilization: Dict[str, List],
    hit_deadline_percentages: Dict[str, List],
    ac_helpers: List[AC_Helper],
    perf_stats_df: DataFrame = DataFrame(),
):
    final_obj = {
        "system_utilization": system_utilization,
        "hit_deadline_percentages": hit_deadline_percentages,
        "ac_helpers": ac_helpers,
        "perf_stats_df": perf_stats_df,
    }
    with open(file_name, "wb") as test_results_file:
        pickle.dump(final_obj, test_results_file)


def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_thread_ids(process_id: int):
    process = psutil.Process(process_id)
    return [thread.id for thread in process.threads()]


def sampling_loop(
    system_utilization: Dict[str, List],
    hit_deadline_percentages: Dict[str, List],
    admission_controller_port: int,
    ac_helpers: List[AC_Helper],
    min_samples: int,
    max_samples: int,
    sampling_period: int,
    sample_start: int = 0,
):

    i = sample_start

    while True:
        if len(ac_helpers) == 0:
            return False

        # Mesaure system utilization using fast api endpoint every sampling period seconds
        start = timer()

        try:
            system_data_response = requests.get(
                f"http://localhost:{admission_controller_port}/stats/system"
            ).json()
            time_stamp = float(system_data_response["time_stamp"])
            for key, value in system_data_response.items():
                if key == "model_inference_stats":
                    model_inference_stats = value
                    for ac_helper in ac_helpers:
                        ac_helper.update_tritonserver_inference_stats(
                            time_stamp,
                            model_inference_stats[ac_helper.model_name],
                        )
                elif key == "model_inference_stats_projected":
                    model_inference_stats_projected = value
                else:
                    system_utilization[key].append([time_stamp, value])
        except Exception as e:
            assert False, (
                "Failed to fetch stats from admission controller with"
                f" error: {e}"
            )

        # Update metrics for plotting
        relative_std_errors = []
        for ac_helper in ac_helpers:
            missed_deadline_percentage = ac_helper.miss_rate * 100
            hit_deadline_percentages[
                f"{ac_helper.model_name}-{ac_helper.client_id}"
            ].append([time_stamp, 100.0 - missed_deadline_percentage])
            relative_std_errors.append(ac_helper.rel_std_err)

        try:
            print_system_metric_table(time_stamp, system_data_response)
            print_inference_stats_table(
                ac_helpers, model_inference_stats_projected
            )
        except Exception as e:
            logging.warning(
                f"Failed to print metrics stables with {type(e)}: {e}"
            )

        # Break loop when max samples reached or all latency measurements are stable
        stable = len([x for x in relative_std_errors if x < 5]) == len(
            relative_std_errors
        ) and len(relative_std_errors)
        if i >= sample_start + max_samples:
            logging.warning(
                f"Max samples: {max_samples} collected, measurements did not"
                " become stable"
            )
            return False

        if stable and i > sample_start + min_samples:
            logging.info("All measurements stable, ending sampling loop")
            return True

        i += 1

        elapsed = timer() - start
        if elapsed > sampling_period:
            logging.warning(
                f"Can't sample fast enough period: {sampling_period}, loop:"
                f" {elapsed}"
            )
        else:
            sleep(sampling_period - elapsed)


def print_system_metric_table(time_stamp: float, system_data_response: Dict):
    system_metrics_table_headers = [
        "time (s)",
        "sys cpu %",
        "trt cpu %",
        "trt proj cpu %",
        "trt gpu %",
        "trt mem %",
        "trt proj mem %",
        "trt mem uss",
        "trt proj mem uss",
    ]

    system_metrics_table = []

    system_metrics_table.append(
        [
            f"{datetime.fromtimestamp(time_stamp)}",
            f"{system_data_response['system_cpu_util']}",
            f"{system_data_response['triton_cpu_util']}",
            f"{system_data_response['projected_cpu_util']}",
            f"{system_data_response['gpu_util']}",
            f"{system_data_response['triton_memory_util']}",
            f"{system_data_response['projected_triton_memory_util']}",
            f"{psutil._common.bytes2human(system_data_response['triton_memory_uss'])}",
            f"{psutil._common.bytes2human(system_data_response['projected_used_mem'])}",
        ]
    )
    logging.info(
        f"\n{tabulate(system_metrics_table,headers=system_metrics_table_headers)}"
    )


def print_inference_stats_table(
    ac_helpers: List[AC_Helper],
    model_inference_stats_projected: Dict,
):

    # Capture model specific data
    table = []
    headers = [
        "model",
        "type",
        "client",
        "inst",
        "thr",
        "s if cnt",
        "s exec cnt",
        "c if cnt",
        "miss",
        "dl misses",
        "hit %",
        "p",
        "q",
        "cmpt_ipt",
        "cmpt_inf",
        "cmpt_opt",
        "s pj p99 l",
        "s avg l",
        "s rol l",
        "c rol l",
        "slowdown",
        "c 99p l",
        "c stddev l",
        "c seom",
        "c rel err",
        "c avg l",
        "c min l",
        "c max l",
        "avg miss l",
    ]
    for ac_helper in ac_helpers:
        if (
            ac_helper.inference_count > 0
            and ac_helper.server_success_count > 0
        ):
            try:
                instances = str(
                    ac_helper._model_config_dict["instance_group"][0]["count"]
                ) + str(
                    str(
                        ac_helper._model_config_dict["instance_group"][0][
                            "kind"
                        ]
                    ).replace("KIND_", "")
                )
            except KeyError:
                instances = "1/CPU"

            missed_deadline_percentage = ac_helper.miss_rate * 100
            table.append(
                [
                    f"{ac_helper.model_name}",
                    f"{ac_helper.model_type}",
                    f"{ac_helper.client_id}",
                    f"{instances}",
                    f"{ac_helper.inference_count}",
                    f"{ac_helper.server_execution_count}",
                    f"{ac_helper.inference_count}",
                    f"{ac_helper.miss_events}",
                    f"{ac_helper.missed_deadlines}",
                    f"{100.0 - missed_deadline_percentage:.1f}",
                    f"{ac_helper.period * 1e3}",
                    f"{ac_helper.server_queue_time:.3f}",
                    f"{ac_helper.server_compute_input:.3f}",
                    f"{ac_helper.server_compute_infer:.3f}",
                    f"{ac_helper.server_compute_output:.3f}",
                    f"{model_inference_stats_projected[ac_helper.model_name]['perf_latency_p99']}",
                    f"{ac_helper.server_average_latency:.1f}",
                    f"{ac_helper.server_rolling_latency:.1f}"
                    if ac_helper.server_rolling_latency
                    else "N/A",
                    f"{ac_helper.rolling_avg_latency * 1e3:.3f}",
                    f"{ac_helper.slowdown:.3f}",
                    f"{ac_helper.p99_latency * 1e3:.3f}",
                    f"{ac_helper.stddev_latency * 1e3:.3f}",
                    f"{ac_helper.seom * 1e3:.3f}",
                    f"{ac_helper.rel_std_err:.3f}",
                    f"{ac_helper.avg_latency * 1e3:.3f}",
                    f"{ac_helper.min_latency * 1e3:.3f}",
                    f"{ac_helper.max_latency * 1e3:.3f}",
                    f"{ac_helper.avg_miss_latency * 1e3}",
                ]
            )

            insert_index = 4

            # Handle thread count
            if ac_helper.model_type == "tf":
                try:
                    threads = ac_helper._model_config_dict["parameters"][
                        "TF_NUM_INTRA_THREADS"
                    ]["string_value"]
                except KeyError:
                    threads = multiprocessing.cpu_count()
            elif ac_helper.model_type == "tflite":
                try:
                    threads = ac_helper._model_config_dict["parameters"][
                        "tflite_num_threads"
                    ]["string_value"]
                except KeyError:
                    threads = multiprocessing.cpu_count()
            table[-1].insert(insert_index, str(threads))
            insert_index += 1

            try:
                accelerator = ac_helper._model_config_dict["optimization"][
                    "execution_accelerators"
                ]["cpu_execution_accelerator"][0]["name"]
            except KeyError as e:
                accelerator = "n/a"
            table[-1].insert(insert_index, accelerator)
            if "acc" not in headers:
                headers.insert(insert_index, "acc")
            insert_index += 1

            try:
                cpu_accel_threads = ac_helper._model_config_dict[
                    "optimization"
                ]["execution_accelerators"]["cpu_execution_accelerator"][0][
                    "parameters"
                ][
                    "num_threads"
                ]
            except KeyError as e:
                if "num_threads" in str(e):
                    cpu_accel_threads = multiprocessing.cpu_count()
                else:
                    cpu_accel_threads = "n/a"
            table[-1].insert(insert_index, str(cpu_accel_threads))
            if "acc thr" not in headers:
                headers.insert(insert_index, "acc thr")
            insert_index += 1

            try:
                fast_math = ac_helper._model_config_dict["optimization"][
                    "execution_accelerators"
                ]["cpu_execution_accelerator"][0]["parameters"][
                    "fast_math_enabled"
                ]
            except:
                fast_math = "n/a"
            table[-1].insert(insert_index, str(fast_math))
            if "fmath" not in headers:
                headers.insert(insert_index, "fmath")
            insert_index += 1

    logging.info(f"\n{tabulate(table,headers=headers)}")
