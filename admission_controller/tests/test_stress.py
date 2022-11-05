# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pathlib
import pprint
import uuid
from collections import defaultdict
from time import sleep
from typing import List

import pandas as pd
import pytest
import requests  # type: ignore

from tests.helpers.ac_perf_analyzer.ac_helper import AC_Helper
from tests.helpers.helpers import *
from tests.helpers.plotting import plot_results_default


@pytest.mark.parametrize(
    "models",
    [
        pytest.param(
            ("inception_graphdef", "ambient_sound_clf"),
            marks=(
                pytest.mark.tf,
                pytest.mark.large,
                pytest.mark.small,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ("vggish", "ambient_sound_clf"),
            marks=(
                pytest.mark.tf,
                pytest.mark.small,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ["efficientnet_quant"] * 2,
            marks=(pytest.mark.tflite, pytest.mark.medium, pytest.mark.quant),
        ),
        pytest.param(
            ["efficientnet_quant"] * 3,
            marks=(pytest.mark.tflite, pytest.mark.medium, pytest.mark.quant),
        ),
        pytest.param(
            ["efficientnet_quant"] * 4,
            marks=(pytest.mark.tflite, pytest.mark.medium, pytest.mark.quant),
        ),
        pytest.param(
            ["efficientnet_quant"] * 5,
            marks=(pytest.mark.tflite, pytest.mark.medium, pytest.mark.quant),
        ),
        pytest.param(
            ("efficientnet_quant", "inceptionv3"),
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.quant,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ("vggish", "ambient_sound_clf", "inception_graphdef"),
            marks=(
                pytest.mark.tf,
                pytest.mark.small,
                pytest.mark.large,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ("inceptionv3", "mobilenet_v1_1.0_224"),
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.small,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ("inceptionv3", "resnet_v2_101_fp32"),
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ["inceptionv3", "resnet_v2_101_fp32"] * 2,
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ["inceptionv3", "resnet_v2_101_fp32"] * 4,
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ("resnet_v2_101_fp32", "mobilenet_v1_1.0_224"),
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.small,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            ("resnet_v2_101_fp32", "mobilenet_v1_1.0_224", "inceptionv3"),
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.small,
                pytest.mark.float,
            ),
        ),
        pytest.param(
            (
                "resnet_v2_101_fp32",
                "mobilenet_v1_1.0_224",
                "inceptionv3",
                "deit_float32",
                "efficientnet_quant",
            ),
            marks=(
                pytest.mark.tflite,
                pytest.mark.medium,
                pytest.mark.small,
                pytest.mark.float,
                pytest.mark.quant,
            ),
        ),
        pytest.param(
            ("resnet_v2_101_fp32", "ambient_sound_clf", "inceptionv3"),
            marks=(
                pytest.mark.mixed,
                pytest.mark.medium,
                pytest.mark.small,
                pytest.mark.float,
            ),
        ),
    ],
)
def test_stress_existing_config(
    admission_controller,
    sampling_period,
    min_capture_duration,
    max_capture_duration,
    models,
    slack,
    threads,
    instances,
    batch_size,
    load_method,
    platform,
    unique_models,
    write_inference_output,
    triton_port_str,
    regular_sched,
    perf_stats,
    perf_stats_tracker,
    scaling_factor,
    triton_pid,
    plot,
    request,
):

    adm_port = admission_controller.ports["2520/tcp"][0]

    # assert container healthy
    assert (
        requests.get(f"http://localhost:{adm_port}/health").status_code == 200
    )

    ac_helpers: List[AC_Helper] = []
    model_counts = defaultdict(int)
    model_name_pairs = []

    base_triton_thread_ids = get_thread_ids(triton_pid)
    logging.info(f"triton base threads: {len(base_triton_thread_ids)}")

    hit_deadline_percentages = defaultdict(list)
    system_utilization = defaultdict(list)

    # Generate model name and model load name pairs to account for unique models option
    for model_name in models:
        # If unique model_names, tell admission controller to treat same model file
        # as different model by using a unique model load name
        if unique_models:
            model_load_name = model_name + "-" + str(uuid.uuid4())[:8]
        else:
            model_load_name = model_name

        model_counts[model_load_name] += 1

        model_name_pairs.append(
            (model_name, model_load_name),
        )

    for model_name_pair in model_name_pairs:
        model_name = model_name_pair[0]
        model_load_name = model_name_pair[1]
        # Create AC_Helper instance pointing at admission_controller container instance
        ac_helper = AC_Helper(
            host="localhost",
            port=adm_port,
            model_name=model_load_name,
            model_path=f"{pathlib.Path(__file__).parent.resolve()}/triton_model_repo/{model_name}",
            triton_url=(
                f"localhost:{admission_controller.ports[triton_port_str][0]}"
            ),
            load_type="existing",
            method=load_method,
            write_inference_output=write_inference_output,
            platform_name=platform,
            instances=instances,
            threads=threads,
            cpu_accelerator_num_threads=threads,
            request_batch_size=batch_size,
            concurrency=model_counts[model_load_name],
            scaling_factor=scaling_factor,
            regular_sched=regular_sched,
        )

        # Compute a target throughput based on the loaded metrics
        ac_helper.throughput_constraint = 1 / (
            (ac_helper.projected_p99_latency * scaling_factor / 1e3)
            * (1 + slack)
        )
        logging.info(
            f"Model {model_load_name} throughput set to"
            f" {ac_helper.throughput_constraint} period is"
            f" {ac_helper.period}"
        )

        # Call to load endpoint of admission controller for current model with random request rate
        # This call causes other inference loops to miss deadlines, so we tell them to stop before loading
        # the model, then resume afterwards
        for helper in ac_helpers:
            while not (helper.inference_halted):
                helper.halt_infer_process()
                logging.info("waiting for helper to halt")
                sleep(1)

        # Upload model file and triton model configuration to upload endpoint of model controller
        res = ac_helper.upload_model(with_profile_data=True)
        assert res.status_code in [
            201,
            303,
        ], (
            f"Failed to upload model: {model_load_name}\n Code:"
            f" {res.status_code} Response: {res.text}"
        )
        logging.info(
            f"Upload model {model_load_name} success with response {res.text}"
        )
        res = ac_helper.load_model()

        # Did model fit?
        assert res.status_code in [
            201,
            303,
            409,
        ], (
            f"Load model {model_load_name} unexpected failure:"
            f" Code: {res.status_code} Response: {res.text}"
        )
        if res.status_code == 409:
            logging.warning(
                f"Failed to load model {model_load_name}. Response: {res.text}"
            )
            break
        else:
            logging.info(
                f"Model {model_load_name} loaded with response"
                f" {pprint.pformat(res.text)}"
            )

        ac_helpers.append(ac_helper)

        # Spin off request generation loop at desired frequency
        ac_helper.create_infer_process(
            update_interval=sampling_period,
        )
        logging.info("Starting all infer processes")
        for helper in ac_helpers:
            helper.start_infer_process()

        # Assign threads associated with the inference for the current model
        # Only tracks the first model in the list
        if perf_stats and len(ac_helpers) == 1:
            retries = 0
            while retries < 5:
                current_triton_thread_ids = get_thread_ids(triton_pid)
                model_thread_ids_set = set(current_triton_thread_ids) - set(
                    base_triton_thread_ids
                )
                ac_helper.model_threads = list(model_thread_ids_set)
                if len(model_thread_ids_set) > 0:
                    sleep(0.5)
                    break
                retries += 1
            assert (
                retries != 5
            ), f"Could not identify threads for model {ac_helper.model_name}"

            logging.info(
                f"Threads belonging to {ac_helper.model_name}:"
                f" {ac_helper.model_threads}"
            )
            perf_stats_tracker.profile_new_thread_group(
                ac_helper.model_name, ac_helper.model_threads
            )

        # Loop for the sampling duration period of time
        min_samples = int(min_capture_duration / sampling_period)
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

    if perf_stats:
        logging.info("Perf Stats Results")
        perf_stats_df = perf_stats_tracker.stop_stats_profile_threads()
    else:
        perf_stats_df = pd.DataFrame()

    # Kill the inference loop processes
    for ac_helper in ac_helpers:
        ac_helper.stop_infer_process()

    # Normalize the result data
    normalize_timeseries_data(
        system_utilization, hit_deadline_percentages, ac_helpers, perf_stats_df
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
        perf_stats_df,
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
            perf_stats_df,
        )

    logging.info("Exiting")

    logging.info("AdmissionController logs")
    logging.info(admission_controller.logs())
