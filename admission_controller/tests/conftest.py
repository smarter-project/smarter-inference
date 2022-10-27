# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import math
import multiprocessing
import os
import pathlib
import random
import socket
from time import sleep
from typing import Dict, List

import docker
import psutil
import pytest
from admission_controller.app.config_generator.model_config_generator import (
    ModelConfigGenerator,
)
from admission_controller.app.constants import cpu_list_from_cpu_str
from admission_controller.app.deadlines import Deadlines
from admission_controller.app.model import ModelType
from cpuset.cset import CpuSet
from pandas import DataFrame
from pytest_docker_tools import build, container, fxtr

from tests.helpers.helpers import powerset
from tests.helpers.perf_stats import PerfStats

admission_controller_image = build(
    path=f"{pathlib.Path(__file__).parent.resolve()}/../../",
    dockerfile="{dockerfile}",
    buildargs={"TRITON_BASE_IMAGE": "{triton_base_image}"},
)

admission_controller = container(
    image="{admission_controller_image.id}",
    ports=fxtr("adm_ports"),
    scope="function",
    environment=fxtr("adm_environment"),
    cap_add=["SYS_NICE"],
    volumes=fxtr("volumes"),
    ulimits=[docker.types.Ulimit(name="rtprio", soft=50, hard=50)],
)


@pytest.fixture(
    scope="session",
)
def triton_port_str(request):
    if not (request.config.getoption("no_nginx")):
        return "2521/tcp"
    else:
        return "8000/tcp"


@pytest.fixture(
    scope="function",
)
def perf_stats_tracker(triton_pid, perf_events, perf_sampling_period):
    logging.info("Collecting perf stats")
    perf_stats_tracker = PerfStats(
        triton_pid,
        events=perf_events.split(","),
        sampling_interval_ms=perf_sampling_period,
    )
    perf_stats_tracker.start_stats_profile()
    return perf_stats_tracker


@pytest.fixture(scope="session")
def adm_ports(request):
    base_ports = {
        "2520/tcp": None,
        "2521/tcp": None,
        "2522/tcp": None,
    }
    if request.config.getoption("no_nginx"):
        base_ports["8000/tcp"] = None
    return base_ports


@pytest.fixture(
    scope="session",
)
def adm_environment(request, nginx_logging, gpus):
    env = {
        "NO_ENFORCE": "1"
        if request.config.getoption("no_enforce_admission")
        else "",
        "SCALING_FACTOR": "1"
        if request.config.getoption("scaling_factor")
        else "",
        "NGINX_LOGGING": "1" if nginx_logging else "",
        "GPUS": gpus,
        "TRITON_CPUS": request.config.getoption("cpus"),
        "NO_NGINX": "1" if request.config.getoption("no_nginx") else "",
        "NO_NGINX_RATELIMIT": "1"
        if request.config.getoption("no_rate_limit")
        else "",
        "LOG_LEVEL": request.config.getoption("log"),
        "MULTI_SERVER": "1"
        if request.config.getoption("multi_server")
        else "",
        "SHIELD": "1" if request.config.getoption("shield") else "",
        "PRIORITY_ASSIGNMENT": request.config.getoption(
            "pri_assignment"
        ).upper(),
    }
    if request.config.getoption("nice"):
        env["SCHED"] = "NICE"
    elif request.config.getoption("deadline"):
        env["SCHED"] = "DEADLINE"
    elif request.config.getoption("rr"):
        env["SCHED"] = "RR"
    else:
        env["SCHED"] = ""

    return env


@pytest.fixture(
    scope="session",
)
def nginx_logging(request):
    if request.config.getoption("no_nginx"):
        return False
    else:
        return request.config.getoption("nginx_logging")


@pytest.fixture(
    scope="session",
)
def volumes(request):
    volumes = (
        {"/usr/bin/tegrastats": {"bind": "/usr/bin/tegrastats", "mode": "rw"}}
        if request.config.getoption("jetson")
        else {}
    )
    if request.config.getoption("shield"):
        volumes["/sys/fs/cgroup"] = {"bind": "/sys/fs/cgroup", "mode": "rw"}
    return volumes


@pytest.fixture(
    scope="function",
)
def triton_pid(admission_controller):
    # Find tritonserver process pid
    triton_pid = None

    # Attempt to find the tritonserver process 5 times
    for i in range(5):
        triton_server_pids = []
        sleep(1)
        for proc in psutil.process_iter():
            if "tritonserver" in proc.name():
                triton_pid = proc.pid
                triton_server_pids.append(proc.pid)

        if len(triton_server_pids) > 1:
            logging.error(
                "Too many tritonserver processes active on system:"
                f" {triton_server_pids}"
            )
            continue
        elif not (len(triton_server_pids)):
            logging.error("No triton found on system")
            continue

        if triton_pid:
            return triton_pid

    return None


@pytest.fixture(
    scope="session",
)
def num_cpus(request):
    if request.config.getoption("shield"):
        cpuset = CpuSet()
        cpuset.read_cpuset("/user")
        return len(cpu_list_from_cpu_str(cpuset.getcpus()))
    else:
        return len(cpu_list_from_cpu_str(request.config.getoption("cpus")))


def thread_counts():
    cpus = multiprocessing.cpu_count()
    thread_max_log = math.log2(cpus)
    thread_count = [
        2**j
        for j in range(
            int(thread_max_log + (1 if thread_max_log.is_integer() else 2)),
        )
    ]
    if thread_count[-1] > cpus:
        thread_count[-1] = cpus

    return thread_count


def find_cpu_percentage_pairs(
    model_names: List[str],
    platform: str,
    num_available_cpus: int,
    bucket_sizes: int,
    no_max_threads: bool = False,
    no_oversub: bool = False,
):
    dfs: Dict[str, DataFrame] = {}
    # Populate dataframes with perf data
    for model_name in model_names:
        metrics_file_name = os.path.join(
            os.path.dirname(__file__),
            "triton_model_repo",
            model_name,
            f"metrics-model-inference-{platform}.csv",
        )
        _, _, dfs[model_name] = ModelConfigGenerator.model_analyzer_csv_to_df(
            ModelType.tflite, metrics_file_name
        )
        # Remove inference latencies over 5 seconds
        dfs[model_name] = dfs[model_name][
            dfs[model_name]["p99 Latency (ms)"] < 5000.0
        ]
        dfs[model_name] = dfs[model_name][
            (dfs[model_name]["Concurrency"] == 1)
            & (dfs[model_name]["Batch"] == 1)
            & (dfs[model_name]["CPU"] == 1)
        ]
        if no_max_threads:
            dfs[model_name] = dfs[model_name][
                (
                    dfs[model_name]["backend_parameter/tflite_num_threads"]
                    < num_available_cpus
                )
                & (
                    dfs[model_name]["cpu_accelerator_num_threads"]
                    < num_available_cpus
                )
            ]
        dfs[model_name]["name"] = model_name

        # Period in seconds
        dfs[model_name]["period"] = (
            dfs[model_name]["p99 Latency (ms)"]
            / 1000
            * (1 + max(random.random() * 1, 0.2))
        )

        assert not (
            dfs[model_name].empty
        ), "Could not find perf entries matching requirements"

    # Generate sets of model rows, and get cpu util values
    model_combinations = powerset(model_names)
    model_combinations.pop(0)

    test_configs = []

    # Create buckets 0-90 in steps of 10
    buckets: Dict = {x: [] for x in range(0, 100, 10)}

    for bucket in buckets:
        logging.info(f"Bucket is {bucket}")
        # Fill every bucket with bucket_sizes test configs
        filled = 0
        while filled < bucket_sizes:
            model_combination = random.choice(model_combinations)
            # Randomly select rows from data
            rows = []
            for model_name in model_combination:
                rows.append(dfs[model_name].sample())

            model_infos: List[Deadlines.ModelInfo] = []
            for row in rows:
                model_infos.append(
                    Deadlines.ModelInfo(
                        name=row.iloc[0]["name"],
                        period=row.iloc[0]["period"],
                        latency=(row.iloc[0]["p99 Latency (ms)"] / 1000),
                        max_model_cpu_util=row.iloc[0]["CPU Utilization (%)"],
                    )
                )
            # If no oversub passed, find combinations which don't result in threads > cores
            if no_oversub:
                threads = 0
                for row in rows:
                    threads += max(
                        row.iloc[0]["backend_parameter/tflite_num_threads"],
                        row.iloc[0]["cpu_accelerator_num_threads"],
                    )
                if threads > num_available_cpus:
                    continue

            # Enter loop to modify the requests to generate the target cpu util for current bucket
            cpu_util = -1.0
            i = 0
            while i < 15:
                # Compute the cpu util for the given combination
                hyperperiod = Deadlines.get_hyperperiod(
                    [model_info.period for model_info in model_infos]
                )
                cpu_util = sum(
                    [
                        (model.latency * model.max_model_cpu_util)
                        * (hyperperiod / model.period)
                        for model in model_infos
                    ]
                ) / (hyperperiod * num_available_cpus)

                if cpu_util >= bucket and cpu_util < (bucket + 10):
                    # If cpu util in range, get deadline misses and append the test config to the list
                    logging.info(
                        f"Found config for bucket {bucket}, cpu_util:"
                        f" {cpu_util}"
                    )
                    (_, cpu_util,) = Deadlines.predict_deadline_misses(
                        model_infos, num_available_cpus, hyperperiod
                    )
                    test_configs.append((rows, hyperperiod))
                    break
                else:
                    index = random.randint(0, len(model_infos) - 1)

                if cpu_util < bucket:
                    # If cpu util too low, randomly select row and reduce period to increase cpu util
                    if (model_infos[index].period * 0.8) > (
                        model_infos[index].latency
                    ):
                        model_infos[index].period *= 0.8
                else:
                    # Cpu too high increase period to decrease cpu util
                    model_infos[index].period *= 1.2

                i += 1
            if i < 15:
                filled += 1
    return test_configs


@pytest.fixture(
    scope="session",
)
def dockerfile(request):
    return (
        "Dockerfile.jetson"
        if request.config.getoption("jetson")
        else "Dockerfile"
    )


@pytest.fixture(
    scope="session",
)
def gpus(request):
    return "0" if request.config.getoption("jetson") else ""


def platform():
    """
    Determine the platform based on the hostname

    Returns:
        str: string hostname or empty string if unknown platform
    """
    hostname = socket.gethostname()
    if hostname == "smarter-server":
        return "smarter"
    elif hostname == "cn18":
        return "thunderx2"
    elif hostname == "josmin01-xavier":
        return "xavier"
    elif hostname == "josmin01-rpi4-2":
        return "rpi"
    elif hostname == "josmin01-npi":
        return "npi"
    elif hostname == "jish-honeycomb":
        return "honey"
    else:
        return ""


def pytest_addoption(parser):
    """
    Adds the program option 'url' to pytest
    """
    parser.addoption(
        "--log", action="store", default="INFO", help="set logging level"
    )
    parser.addoption(
        "--host",
        action="store",
        default="localhost",
        required=False,
        help="admission-controller URL. Default is localhost:2520.",
    )
    parser.addoption(
        "--model-repo-path",
        action="store",
        default="test_model_repo",
        required=False,
        help="Path to top level of triton model repository",
    )
    parser.addoption(
        "--triton-base-image",
        action="store",
        default="tritonserver:latest",
        required=False,
        help="Image name for triton used in admission controller",
    )
    parser.addoption(
        "--backend-directory",
        action="store",
        default="/tmp/citritonbuild/opt/tritonserver/backends",
        required=False,
        help="Path to triton backends",
    )
    parser.addoption(
        "--sampling-period",
        action="store",
        default=1,
        type=int,
        required=False,
        help="Frequency to sample statistics from admission controller",
    )
    parser.addoption(
        "--min-capture-duration",
        action="store",
        default=10,
        type=int,
        required=False,
        help="Min length to capture data for each test",
    )
    parser.addoption(
        "--max-capture-duration",
        action="store",
        default=60,
        type=int,
        required=False,
        help="Max length to capture data for each test",
    )
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        required=False,
        help="Plot results of test",
    )
    parser.addoption(
        "--no-enforce-admission",
        action="store_true",
        default=False,
        required=False,
        help="Run tests without admission enforcement",
    )
    parser.addoption(
        "--no-rate-limit",
        action="store_true",
        default=False,
        required=False,
        help="Run tests without nginx rate limiting",
    )
    parser.addoption(
        "--no-nginx",
        action="store_true",
        default=False,
        required=False,
        help="Bypass NGINX frontend on Triton",
    )
    parser.addoption(
        "--nginx-logging",
        action="store_true",
        default=False,
        required=False,
        help="Write nginx error log",
    )
    parser.addoption(
        "--regular-sched",
        action="store_true",
        default=False,
        required=False,
        help=(
            "If client misses deadline, will wait to synchronize sending of"
            " next request based on original schedule"
        ),
    )
    parser.addoption(
        "--perf-stats",
        action="store_true",
        default=False,
        required=False,
        help="Use perf to gather perf stats",
    )
    parser.addoption(
        "--perf-sampling-period",
        action="store",
        type=int,
        required=False,
        help="Frequency to sample perf statistics in ms",
    )
    parser.addoption(
        "--perf-events",
        action="store",
        default=(
            "L1-dcache-loads,L1-dcache-load-misses,"
            "L1-dcache-stores,LLC-loads,LLC-load-misses,"
            "LLC-stores,cache-misses,cache-references"
        ),
        required=False,
        help="Perf events to capture",
    )
    parser.addoption(
        "--jetson",
        action="store_true",
        default=False,
        required=False,
        help="Run tests on Jetson",
    )
    parser.addoption(
        "--unique-models",
        action="store_true",
        default=False,
        required=False,
        help=(
            "Instruct admission controller to allocate a single model per"
            " client"
        ),
    )
    parser.addoption(
        "--write-inference-output",
        action="store_true",
        default=False,
        required=False,
        help="Write inference output to file",
    )
    parser.addoption(
        "--load-method",
        default="lookup",
        choices=["lookup"],
        required=False,
        help="Method to use to handle model configs",
    )
    parser.addoption(
        "--threads",
        nargs="+",
        type=int,
        default=thread_counts(),
        required=False,
        help="Threads to test with",
    )
    parser.addoption(
        "--slack",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        required=False,
        help="Model names used in accuracy test to generate tests",
    )
    parser.addoption(
        "--model-names",
        nargs="+",
        type=str,
        default=[
            "inceptionv3",
            "efficientnet_quant",
        ],
        required=False,
        help="Slack in deadlines vs measured latency",
    )
    parser.addoption(
        "--scaling-factor",
        action="store",
        type=float,
        default=1.0,
        required=False,
        help="Scaling based on perf data clock speed to test clock speed",
    )
    parser.addoption(
        "--cpus",
        action="store",
        default="",
        required=False,
        help="CPUs to use for admission controller",
    )
    parser.addoption(
        "--bucket-sizes",
        action="store",
        default=5,
        type=int,
        required=False,
        help="Bucket sizes to generate acc tests with",
    )
    parser.addoption(
        "--no-max-threads",
        action="store_true",
        default=False,
        required=False,
        help=(
            "Do not generate tests with model thread counts equal to total"
            " cores in system"
        ),
    )
    parser.addoption(
        "--no-oversub",
        action="store_true",
        default=False,
        required=False,
        help=(
            "Do not generate tests with model thread counts greater to total"
            " cores in system"
        ),
    )
    parser.addoption(
        "--multi-server",
        action="store_true",
        default=False,
        required=False,
        help="Create an instance of triton for each model",
    )
    parser.addoption(
        "--deadline",
        action="store_true",
        default=False,
        required=False,
        help="Use linux deadline sched class",
    )
    parser.addoption(
        "--rr",
        action="store_true",
        default=False,
        required=False,
        help="Use linux round robin sched class",
    )
    parser.addoption(
        "--nice",
        action="store_true",
        default=False,
        required=False,
        help="Use linux nice to set priority",
    )
    parser.addoption(
        "--shield",
        action="store_true",
        default=False,
        required=False,
        help="Run tritonserver in cpu shield",
    )
    parser.addoption(
        "--pri-assignment",
        default="throughput",
        choices=["throughput", "slack"],
        required=False,
        help="Method to assign priorities to model inference threads",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if "host" in metafunc.fixturenames:
        metafunc.parametrize(
            "host", [metafunc.config.getoption("host")], scope="module"
        )
    if "model_repo_path" in metafunc.fixturenames:
        metafunc.parametrize(
            "model_repo_path",
            [metafunc.config.getoption("model_repo_path")],
            scope="module",
        )
    if "backend_directory" in metafunc.fixturenames:
        metafunc.parametrize(
            "backend_directory",
            [metafunc.config.getoption("backend_directory")],
            scope="module",
        )
    if "sampling_period" in metafunc.fixturenames:
        metafunc.parametrize(
            "sampling_period",
            [metafunc.config.getoption("sampling_period")],
            scope="module",
        )
    if "min_capture_duration" in metafunc.fixturenames:
        metafunc.parametrize(
            "min_capture_duration",
            [metafunc.config.getoption("min_capture_duration")],
            scope="module",
        )
    if "max_capture_duration" in metafunc.fixturenames:
        metafunc.parametrize(
            "max_capture_duration",
            [metafunc.config.getoption("max_capture_duration")],
            scope="module",
        )
    if "no_enforce_admission" in metafunc.fixturenames:
        metafunc.parametrize(
            "no_enforce_admission",
            [metafunc.config.getoption("no_enforce_admission")],
            scope="module",
            ids=[
                "no_enforce_admission"
                if metafunc.config.getoption("no_enforce_admission")
                else "enforce_admission"
            ],
        )
    if "triton_base_image" in metafunc.fixturenames:
        metafunc.parametrize(
            "triton_base_image",
            [metafunc.config.getoption("triton_base_image")],
            scope="session",
        )

    if "regular_sched" in metafunc.fixturenames:
        metafunc.parametrize(
            "regular_sched",
            [metafunc.config.getoption("regular_sched")],
            scope="session",
            ids=[
                "regular_sched"
                if metafunc.config.getoption("regular_sched")
                else "irreg_sched"
            ],
        )

    if "perf_stats" in metafunc.fixturenames:
        metafunc.parametrize(
            "perf_stats",
            [metafunc.config.getoption("perf_stats")],
            scope="session",
            ids=[
                "perf_stats"
                if metafunc.config.getoption("perf_stats")
                else None
            ],
        )

    if "perf_sampling_period" in metafunc.fixturenames:
        metafunc.parametrize(
            "perf_sampling_period",
            [metafunc.config.getoption("perf_sampling_period", default=None)],
            scope="module",
            ids=[""],
        )

    if "perf_events" in metafunc.fixturenames:
        metafunc.parametrize(
            "perf_events",
            [metafunc.config.getoption("perf_events")],
            scope="session",
            ids=[""],
        )

    if "plot" in metafunc.fixturenames:
        metafunc.parametrize(
            "plot",
            [metafunc.config.getoption("plot")],
            scope="session",
            ids=[""],
        )

    if "unique_models" in metafunc.fixturenames:
        metafunc.parametrize(
            "unique_models",
            [metafunc.config.getoption("unique_models")],
            scope="module",
            ids=[
                "unique_models"
                if metafunc.config.getoption("unique_models")
                else "shared_models"
            ],
        )
    if "write_inference_output" in metafunc.fixturenames:
        metafunc.parametrize(
            "write_inference_output",
            [metafunc.config.getoption("write_inference_output")],
            scope="module",
            ids=[None],
        )
    if "load_method" in metafunc.fixturenames:
        metafunc.parametrize(
            "load_method",
            [metafunc.config.getoption("load_method")],
            scope="session",
        )
    if "threads" in metafunc.fixturenames:
        metafunc.parametrize(
            "threads",
            metafunc.config.getoption("threads"),
            scope="session",
        )
    if "slack" in metafunc.fixturenames:
        metafunc.parametrize(
            "slack",
            metafunc.config.getoption("slack"),
            scope="session",
        )
    if "model_names" in metafunc.fixturenames:
        metafunc.parametrize(
            "model_names",
            [metafunc.config.getoption("model_names")],
            scope="session",
        )
    if "scaling_factor" in metafunc.fixturenames:
        metafunc.parametrize(
            "scaling_factor",
            [metafunc.config.getoption("scaling_factor")],
            scope="session",
        )
    if "instances" in metafunc.fixturenames:
        instances_list = (
            ["1/GPU", "1/CPU"]
            if metafunc.config.getoption("jetson")
            else ["1/CPU"]
        )
        metafunc.parametrize(
            "instances",
            instances_list,
            ids=[x.replace("/", "") for x in instances_list],
            scope="session",
        )
    if "batch_size" in metafunc.fixturenames:
        metafunc.parametrize(
            "batch_size",
            [1],
            scope="session",
        )
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize(
            "platform",
            [platform()],
            scope="session",
        )
    if "acc_test_config" in metafunc.fixturenames:
        if metafunc.config.getoption("shield"):
            cpuset = CpuSet()
            cpuset.read_cpuset("/user")
            num_avail_cpus = len(cpu_list_from_cpu_str(cpuset.getcpus()))
        else:
            num_avail_cpus = len(
                cpu_list_from_cpu_str(metafunc.config.getoption("cpus"))
            )
        metafunc.parametrize(
            "acc_test_config",
            find_cpu_percentage_pairs(
                metafunc.config.getoption("model_names"),
                platform(),
                num_avail_cpus,
                metafunc.config.getoption("bucket_sizes"),
                metafunc.config.getoption("no_max_threads"),
                metafunc.config.getoption("no_oversub"),
            ),
            scope="session",
        )
