# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import logging
import os
import pathlib
import pprint
import re
from collections import defaultdict
from statistics import mean, median
from time import sleep
from typing import List

from tests.helpers.ac_perf_analyzer.ac_helper import AC_Helper
from tests.helpers.helpers import *
from tests.helpers.plotting import plot_results_default


def test_accuracy(
    admission_controller,
    sampling_period,
    min_capture_duration,
    max_capture_duration,
    load_method,
    platform,
    write_inference_output,
    triton_port_str,
    regular_sched,
    scaling_factor,
    plot,
    request,
    acc_test_config,
    num_cpus,
):
    adm_port = admission_controller.ports["2520/tcp"][0]

    # assert container healthy
    assert (
        requests.get(f"http://localhost:{adm_port}/health").status_code == 200
    )

    hit_deadline_percentages = defaultdict(list)
    system_utilization = defaultdict(list)

    ac_helpers: List[AC_Helper] = []
    rows = acc_test_config[0]
    hyperperiod = acc_test_config[1]
    for row in rows:
        model_name = row.iloc[0]["name"]
        logging.info(model_name)
        # Create AC_Helper instance pointing at admission_controller container instance
        ac_helper = AC_Helper(
            host="localhost",
            port=adm_port,
            model_name=model_name,
            model_path=f"{pathlib.Path(__file__).parent.resolve()}/triton_model_repo/{model_name}",
            triton_url=(
                f"localhost:{admission_controller.ports[triton_port_str][0]}"
            ),
            load_type="existing",
            method=load_method,
            write_inference_output=write_inference_output,
            platform_name=platform,
            threads=row.iloc[0]["backend_parameter/tflite_num_threads"],
            request_batch_size=1,
            concurrency=1,
            scaling_factor=scaling_factor,
            regular_sched=regular_sched,
            cpu_accelerator=row.iloc[0]["cpu_accelerator"],
            cpu_accelerator_num_threads=row.iloc[0][
                "cpu_accelerator_num_threads"
            ],
        )

        # Compute a target throughput based on the loaded metrics
        ac_helper.throughput_constraint = 1 / row.iloc[0]["period"]
        logging.info(
            f"Model {model_name} throughput set to"
            f" {ac_helper.throughput_constraint} period is"
            f" {ac_helper.period}"
        )

        # Upload model file and triton model configuration to upload endpoint of model controller
        res = ac_helper.upload_model(with_profile_data=True)
        assert res.status_code in [
            201,
            303,
        ], (
            f"Failed to upload model: {model_name}\n Code:"
            f" {res.status_code} Response: {res.text}"
        )
        logging.info(
            f"Upload model {model_name} success with response {res.text}"
        )

        expected_deadline_misses = 0

        # Halt all existing helpers
        for helper in ac_helpers:
            while not (helper.inference_halted):
                helper.halt_infer_process()
                logging.info("waiting for helper to halt")
                sleep(1)

        res = ac_helper.load_model()

        ac_helpers.append(ac_helper)

        # Did model fit?
        assert res.status_code in [
            201,
            303,
            409,
        ], (
            f"Load model {model_name} unexpected failure:"
            f" Code: {res.status_code} Response: {res.text}"
        )
        if res.status_code == 409:
            assert (
                False
            ), f"Failed to load model {model_name}. Response: {res.text}"
        else:
            logging.info(
                f"Model {model_name} loaded with response"
                f" {pprint.pformat(res.text)}"
            )
            expected_deadline_misses = int(res.json()["deadline_misses"])

        # Spin off request generation loop at desired frequency
        ac_helper.create_infer_process(
            update_interval=sampling_period,
        )
        logging.info("Starting all infer processes")
        for helper in ac_helpers:
            helper.start_infer_process()

    # Loop for the sampling duration period of time
    min_run_len = max(min_capture_duration, hyperperiod * 1.1)
    min_samples = int(min_run_len / sampling_period)
    max_samples = int(max_capture_duration / sampling_period)

    # Loop for the sampling duration period of time
    stable = sampling_loop(
        system_utilization,
        hit_deadline_percentages,
        admission_controller.ports["2520/tcp"][0],
        ac_helpers,
        min_samples,
        max_samples,
        sampling_period,
    )

    assert stable

    # Kill the inference loop processes
    for ac_helper in ac_helpers:
        ac_helper.stop_infer_process()

    # Normalize the result data
    normalize_timeseries_data(
        system_utilization, hit_deadline_percentages, ac_helpers
    )

    test_dir = os.path.dirname(request.module.__file__)

    pkl_filename = os.path.join(
        test_dir, "raw_test_data", f"{request.node.name}.obj".replace("/", "")
    )
    save_run_results(
        pkl_filename,
        system_utilization,
        hit_deadline_percentages,
        ac_helpers,
    )

    if plot:
        logging.info("Plotting Results")

        plot_filename = os.path.join(
            test_dir, "plots", f"{request.node.name}.pdf".replace("/", "")
        )
        plot_results_default(
            system_utilization,
            hit_deadline_percentages,
            ac_helpers,
            max_capture_duration * sampling_period,
            plot_filename,
        )

    total_deadline_misses = sum(
        [ac_helper.missed_deadlines for ac_helper in ac_helpers]
    )

    # Remove digit from end of test name to aggregate test results
    stripped_test_name = re.sub(
        r"(acc_test_config)(\d*)", r"\1", request.node.name
    )
    plot_filename = os.path.join(
        test_dir, "acc_tests", f"{stripped_test_name}.json"
    )

    # If first test, delete existing file if exists
    if re.search(r"acc_test_config(0{1})", request.node.name):
        if os.path.exists(plot_filename):
            os.remove(plot_filename)
            logging.info(f"Deleted file {plot_filename} successfully")
        else:
            logging.info(f"File {plot_filename} does not exist!")
        # Write empty list to json file
        with open(plot_filename, "a") as f:
            json.dump([], f)

    # Update results json file with new entry
    with open(plot_filename, "r") as f:
        results = json.load(f)

        result = {
            "expected_deadline_misses": expected_deadline_misses,
            "total_deadline_misses": total_deadline_misses,
            "projected_cpu_util": float(
                median(
                    [
                        pair[1]
                        for pair in system_utilization["projected_cpu_util"]
                    ]
                )
            ),
            "triton_cpu_util": float(
                mean(
                    [pair[1] for pair in system_utilization["triton_cpu_util"]]
                )
            ),
            "triton_proj_mem": system_utilization["projected_used_mem"][-2][1],
            "triton_memory": system_utilization["triton_memory_uss"][-2][1],
            "hyperperiod": float(hyperperiod),
            "avail_triton_cpus": num_cpus,
        }
        result["model_data"] = []
        for ac_helper in ac_helpers:
            model_data = {}
            model_data["name"] = ac_helper.model_name
            model_data["hit_deadline_rate"] = float(
                100.0 - (ac_helper.miss_rate * 100)
            )
            model_data["missed_deadlines"] = int(ac_helper.missed_deadlines)
            model_data["period"] = float(ac_helper.period)
            model_data["threads"] = int(ac_helper.threads)
            model_data["accel_threads"] = int(
                ac_helper.cpu_accelerator_num_threads
            )
            result["model_data"].append(model_data)

        results.append(result)

    with open(plot_filename, "w") as f:
        json.dump(results, f)

    if expected_deadline_misses > 0:
        assert total_deadline_misses > 0
    elif expected_deadline_misses == 0:
        assert total_deadline_misses == 0

    logging.info("Exiting")

    logging.info("AdmissionController logs")
    logging.info(admission_controller.logs())
