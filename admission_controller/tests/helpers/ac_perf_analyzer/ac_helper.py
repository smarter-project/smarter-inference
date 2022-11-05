# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import glob
import json
import logging
import os
import select
import subprocess
import threading
import uuid
from statistics import mean
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests  # type: ignore
import tritonclient.grpc.model_config_pb2 as mc
from admission_controller.app.config_generator.model_config_generator import (
    ModelConfigGenerator,
)
from admission_controller.app.model import ModelType
from admission_controller.tests.helpers.ac_perf_analyzer.ac_perf_analyzer_exceptions import (
    ACPerfAnalyzerException,
)
from google.protobuf import json_format, text_format  # type: ignore


class AC_Helper:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        model_path: str,
        triton_url: str,
        load_type: str,
        method: str,
        request_batch_size: int = 1,
        client_type: str = "http",
        write_inference_output: bool = False,
        platform_name: str = "",
        instances: str = "1/CPU",
        threads: int = 1,
        concurrency: int = 1,
        scaling_factor: float = 1.0,
        regular_sched: bool = False,
        throughput_objective_weight: int = 1,
        latency_objective_weight: int = 1,
        cpu_accelerator: Optional[str] = None,
        cpu_accelerator_num_threads: Optional[int] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._model_name = model_name
        self._model_path = model_path  # Top level of model in triton repo
        self._load_type = load_type
        self._method = method
        self._triton_url = triton_url
        self._model_file = self._get_model_file()
        self._client_type = client_type
        self._write_inference_output = write_inference_output
        self._platform_name = platform_name
        (
            self._cpu_instances,
            self._gpu_instances,
        ) = ModelConfigGenerator.get_model_instances(instances)
        self._threads = threads
        self._concurrency = concurrency
        self._scaling_factor = scaling_factor
        self._regular_sched = regular_sched
        self._request_batch_size = request_batch_size

        self._throughput_objective_weight = throughput_objective_weight
        self._latency_objective_weight = latency_objective_weight
        self._cpu_accelerator = cpu_accelerator
        self._cpu_accelerator_num_threads = cpu_accelerator_num_threads

        self._init_model_information()

        self._client_id = str(uuid.uuid4())[:8]
        self._miss_rate = 0.0
        self._miss_events = 0
        self._missed_deadlines = 0
        self._inference_count = 0
        self._avg_latency = 0.0
        self._rolling_avg_latency = 0.0
        self._stddev_latency = 0.0
        self._p99_latency = 0.0
        self._avg_miss_latency = 0.0
        self._min_latency = float(1e10)
        self._max_latency = 0.0
        self._total_requests = 0
        self._server_success_count = 0
        self._inference_start_times: List[float] = []
        self._inference_latencies: List[float] = []
        self._miss_inference_start_times: List[float] = []
        self._miss_inference_latencies: List[float] = []
        self._start_time_stamp = 0.0
        self._model_threads: List[int] = []
        self._time_series_tritonserver_columns = [
            "Time",
            "queue_time",
            "compute_input",
            "computer_infer",
            "compute_output",
            "client_overhead",
        ]
        self._time_series_tritonserver_data = pd.DataFrame(
            columns=self._time_series_tritonserver_columns
        )

        self._finished = threading.Event()
        self._halted = False

        self._ov = self.OnlineVariance()
        self._slowdown = 1.0

        if "maxBatchSize" in self._model_config_dict:
            self._max_batch_size = int(self._model_config_dict["maxBatchSize"])
        else:
            self._max_batch_size = 0

        self._server_queue_time = 0.0
        self._server_compute_infer = 0.0
        self._server_compute_input = 0.0
        self._server_compute_output = 0.0
        self._server_execution_count = 0

    class OnlineVariance(object):
        """
        Welford's algorithm computes the sample variance incrementally.
        """

        def __init__(self, iterable=None, ddof=1):
            self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
            if iterable is not None:
                for datum in iterable:
                    self.include(datum)

        def include(self, datum):
            self.n += 1
            self.delta = datum - self.mean
            self.mean += self.delta / self.n
            self.M2 += self.delta * (datum - self.mean)

        @property
        def variance(self):
            return self.M2 / (self.n - self.ddof)

        @property
        def std(self):
            return np.sqrt(self.variance)

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @property
    def miss_rate(self):
        return self._miss_rate

    @property
    def miss_events(self):
        return self._miss_events

    @property
    def missed_deadlines(self):
        return self._missed_deadlines

    @property
    def throughput_constraint(self):
        return self._throughput_constraint

    @throughput_constraint.setter
    def throughput_constraint(self, constraint: float):
        self._throughput_constraint = constraint
        self._period = 1 / self._throughput_constraint

    @property
    def projected_p99_latency(self):
        return self._projected_p99_latency

    @property
    def period(self):
        return self._period

    @property
    def latency_constraint(self):
        return self._latency_constraint

    @latency_constraint.setter
    def latency_constraint(self, constraint: float):
        self._latency_constraint = constraint

    @property
    def inference_count(self):
        return self._inference_count

    @property
    def server_success_count(self):
        return self._server_success_count

    @property
    def server_execution_count(self):
        return self._server_execution_count

    @property
    def server_queue_time(self):
        return self._server_queue_time

    @property
    def server_compute_input(self):
        return self._server_compute_input

    @property
    def server_compute_infer(self):
        return self._server_compute_infer

    @property
    def server_compute_output(self):
        return self._server_compute_output

    @property
    def server_average_latency(self):
        return self._server_average_latency

    @property
    def server_rolling_latency(self):
        return self._server_rolling_latency

    @property
    def time_series_tritonserver_data(self):
        return self._time_series_tritonserver_data

    @property
    def avg_latency(self):
        return self._avg_latency

    @property
    def rolling_avg_latency(self):
        return self._rolling_avg_latency

    @property
    def stddev_latency(self):
        return self._stddev_latency

    @property
    def p99_latency(self):
        return self._p99_latency

    @property
    def min_latency(self):
        return self._min_latency

    @property
    def max_latency(self):
        return self._max_latency

    @property
    def avg_miss_latency(self):
        return self._avg_miss_latency

    @property
    def client_id(self):
        return self._client_id

    @property
    def inference_start_times(self):
        return self._inference_start_times

    @property
    def inference_latencies(self):
        return self._inference_latencies

    @property
    def miss_inference_start_times(self):
        return self._miss_inference_start_times

    @property
    def miss_inference_latencies(self):
        return self._miss_inference_latencies

    @property
    def inference_halted(self):
        return self._halted

    @property
    def start_time_stamp(self):
        return self._start_time_stamp

    @start_time_stamp.setter
    def start_time_stamp(self, time_stamp: float):
        self._start_time_stamp = time_stamp

    @property
    def rel_std_err(self):
        return 100 * self.seom / self._avg_latency

    @property
    def seom(self):
        return self._stddev_latency / np.sqrt(self._inference_count)

    @property
    def model_threads(self):
        return self._model_threads

    @model_threads.setter
    def model_threads(self, model_threads: List[int]):
        self._model_threads = model_threads

    @property
    def cpu_accelerator_num_threads(self):
        return self._cpu_accelerator_num_threads

    @property
    def threads(self):
        return self._threads

    @property
    def slowdown(self):
        return self._slowdown

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_inference_loop"]
        del state["_finished"]
        del state["_missed_deadlines_check"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._inference_loop = None
        self._finished = None
        self._missed_deadlines_check = None

    def inference_latencies_from_timestamps(
        self, start_time_stamp: float, stop_time_stamp: Optional[float]
    ):
        return [
            latency
            for latency, start_time in list(
                zip(self.inference_latencies, self.inference_start_times)
            )
            if (
                (
                    start_time >= start_time_stamp
                    and start_time <= stop_time_stamp
                )
                if stop_time_stamp
                else (start_time >= start_time_stamp)
            )
        ]

    def _get_perf_row(
        self,
        model_perf_file_path: str,
    ):
        _, _, df = ModelConfigGenerator.model_analyzer_csv_to_df(
            self._model_type, model_perf_file_path
        )

        df = df.loc[
            (df["Batch"] == self._request_batch_size)
            & (df["Concurrency"] == self._concurrency)
            & (df["CPU"] == self._cpu_instances)
            & (df["GPU"] == self._gpu_instances)
        ]

        if self._model_type == ModelType.tflite:
            df = df.loc[
                (df["cpu_accelerator"] != "armnn")
                & (df["backend_parameter/tflite_num_threads"] == self._threads)
            ]
            if self._cpu_accelerator:
                df = df.loc[(df["cpu_accelerator"] == self._cpu_accelerator)]
                if self._cpu_accelerator_num_threads:
                    df = df.loc[
                        (
                            df["cpu_accelerator_num_threads"]
                            == self._cpu_accelerator_num_threads
                        )
                    ]
            elif self._cpu_accelerator_num_threads:
                df = df.loc[
                    (
                        (
                            df["cpu_accelerator_num_threads"]
                            == self._cpu_accelerator_num_threads
                        )
                        | (df["cpu_accelerator_num_threads"] == 0)
                    )
                ]

        if self._model_type == ModelType.tf:
            df = df.loc[
                (df["backend_parameter/TF_NUM_INTRA_THREADS"] == self._threads)
                & (
                    df["backend_parameter/TF_NUM_INTER_THREADS"]
                    == self._threads
                )
            ]

        try:
            return df.iloc[0]
        except:
            return pd.DataFrame()

    def update_tritonserver_inference_stats(
        self, time_stamp: float, model_inference_stats: Dict
    ):
        try:
            self._server_execution_count = model_inference_stats[
                "execution_count"
            ]
            self._server_success_count = model_inference_stats[
                "inference_stats"
            ]["success"]["count"]
        except KeyError:
            return
        if self._inference_count > 0 and self._server_success_count > 0:
            # Register triton server metrics for the model
            self._server_queue_time = (
                model_inference_stats["inference_stats"]["queue"]["ns"]
                / 1e6
                / model_inference_stats["inference_stats"]["queue"]["count"]
            )
            self._server_compute_input = (
                model_inference_stats["inference_stats"]["compute_input"]["ns"]
                / 1e6
                / model_inference_stats["inference_stats"]["compute_input"][
                    "count"
                ]
            )
            self._server_compute_infer = (
                model_inference_stats["inference_stats"]["compute_infer"]["ns"]
                / 1e6
                / model_inference_stats["inference_stats"]["compute_infer"][
                    "count"
                ]
            )
            self._server_compute_output = (
                model_inference_stats["inference_stats"]["compute_output"][
                    "ns"
                ]
                / 1e6
                / model_inference_stats["inference_stats"]["compute_output"][
                    "count"
                ]
            )
            self._server_average_latency = (
                model_inference_stats["inference_stats"]["success"]["ns"]
                / 1e6
                / model_inference_stats["inference_stats"]["success"]["count"]
            )
            try:
                self._server_rolling_latency = (
                    model_inference_stats["rolling_latency"] / 1e6
                )
            except KeyError:
                self._server_rolling_latency = None

            if self._server_rolling_latency:
                self._time_series_tritonserver_data = pd.concat(
                    [
                        self._time_series_tritonserver_data,
                        pd.DataFrame(
                            [
                                [
                                    time_stamp,
                                    self._server_queue_time,
                                    self._server_compute_input,
                                    self._server_compute_infer,
                                    self._server_compute_output,
                                    max(
                                        (
                                            self._rolling_avg_latency * 1e3
                                            - self._server_rolling_latency
                                        ),
                                        0,
                                    ),
                                ]
                            ],
                            columns=self._time_series_tritonserver_columns,
                        ),
                    ],
                    ignore_index=True,
                )
                self._slowdown = model_inference_stats["slowdown"]

    def _init_model_information(self):
        """
        Initialize information related to the model depending on the model load type
        """
        if not os.path.exists(self._model_path):
            raise ACPerfAnalyzerException(
                f'Model path "{self._model_path}" specified does not exist.'
            )

        if os.path.isfile(self._model_path):
            raise ACPerfAnalyzerException(
                f'Model output path "{self._model_path}" must be a directory.'
            )

        model_config_path = os.path.join(self._model_path, "config.pbtxt")
        if not os.path.isfile(model_config_path):
            raise ACPerfAnalyzerException(
                f'Path "{model_config_path}" does not exist.'
                " Make sure that you have specified the correct model"
                " repository and model name(s)."
            )

        with open(model_config_path, "r+") as f:
            config_str = f.read()

        protobuf_message = text_format.Parse(config_str, mc.ModelConfig())

        self._model_config_dict = json_format.MessageToDict(protobuf_message)

        # Get the type of model based on the triton model configuration
        if self._model_config_dict.get("platform") == "tensorflow_graphdef":
            self._model_type = ModelType.tf
        elif self._model_config_dict.get("backend") == "armnn_tflite":
            self._model_type = ModelType.tflite
        else:
            raise ACPerfAnalyzerException("Unknown model filetype")

        # Generate model config based on AC Helper parameters
        self._row = self._get_perf_row(
            f"{self._model_path}/metrics-model-inference-{self._platform_name}.csv",
        )

        if self._load_type == "existing":
            if self._row.empty:
                raise ACPerfAnalyzerException("No matching row found")
            ModelConfigGenerator.update_model_config_from_df_row(
                self._model_config_dict,
                self._model_type,
                self._row,
            )
            logging.info(f"Generated Model config: {self._model_config_dict}")
            # Constraints by default are set the to values for the 99th percentiles as found
            # in the profiling data, however this class allows them to be overwritten.
            # TODO: Add in RAM Usage (MB),CPU Utilization (%)
            self.throughput_constraint = (
                float(self._row["Throughput (infer/sec)"])
                / self._scaling_factor
            )
            self._latency_constraint = (
                float(self._row["p99 Latency (ms)"]) * self._scaling_factor
            )
            self._projected_p99_latency = self._latency_constraint

    def _get_model_file(self):
        """
        Returns the path of the model file given the model name
        """
        return glob.glob(f"{self._model_path}/*/*")[0]

    def create_infer_process(
        self,
        update_interval: int = 10,
    ):
        """
        Starts a background process used for repeated inference
        """
        if self._max_batch_size == 0 and self._request_batch_size > 1:
            raise ACPerfAnalyzerException(
                "Batch size must be less than max batch size of 1"
            )
        if self._max_batch_size > 0 and (
            self._request_batch_size > self._max_batch_size
        ):
            raise ACPerfAnalyzerException(
                "Batch size must be less than max batch size of"
                f" {self._max_batch_size}"
            )

        if self._max_batch_size == 0:
            batch_size = None
        else:
            batch_size = self._request_batch_size

        cmd = [
            "python",
            "-u",
            os.path.join(os.path.dirname(__file__), "inference_loop.py"),
            self._client_type,
            self._endpoint_uuid,
            self._client_id,
            self._model_name,
            json.dumps(self._model_config_dict),
            self._triton_url,
            str(self._period),
            str(batch_size),
            str(update_interval),
            str(self._regular_sched),
        ]

        # Start inference loop in subprocess
        self._inference_loop = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )

        # Check for missed deadlines from inference loop in background thread
        self._missed_deadlines_check = threading.Thread(
            target=self.gather_events, daemon=True
        )
        self._missed_deadlines_check.start()

        self._start_time_stamp = time()

    def start_infer_process(self):
        self._inference_loop.stdin.write("start\n")
        self._inference_loop.stdin.flush()
        atexit.register(self.stop_infer_process)

    def halt_infer_process(self):
        self._inference_loop.stdin.write("halt\n")
        self._inference_loop.stdin.flush()

    def stop_infer_process(self):
        self._finished.set()
        if not (self._inference_loop.poll()):
            self._inference_loop.kill()

    def gather_events(self):
        """
        Read from the connections to the periodic inference threads to
        update miss counts in the missed_deadlines dict
        This function may wait forever
        """

        if self._write_inference_output:
            filename = f"inference_output/output-{self._model_name}-{self._client_id}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                self._process_msg_loop(f)
        else:
            self._process_msg_loop()

    def _process_msg_loop(self, f=None):
        try:
            ep = select.epoll()
            ep.register(
                self._inference_loop.stdout.name,
                select.EPOLLIN | select.EPOLLHUP,
            )
            while not (self._inference_loop.poll()):
                evts = ep.poll()
                for fd, evt in evts:
                    if evt & select.EPOLLIN:
                        self._process_output_line(
                            self._inference_loop.stdout.readline().rstrip(), f
                        )
                    if evt & select.EPOLLHUP:
                        for line in self._inference_loop.stdout:
                            self._process_output_line(line.rstrip(), f)
                        ep.unregister(fd)

        except Exception as e:
            logging.error(e)

        finally:
            ep.close()

    def _process_output_line(self, msg: str, f=None):
        if f:
            f.write(f"{msg}\n")
        if "latencies," in msg:
            self._update_client_inference_stats(msg)
        if "halted" in msg:
            self._halted = True
        if "started" in msg:
            self._halted = False

    def _update_client_inference_stats(self, inf_line_str: str):
        inference_data = inf_line_str.split(",")

        # Data comes in as timestamp followed by inference latency in ms
        start_times = [float(i) for i in inference_data[1::2]]
        latencies = [float(i) for i in inference_data[2::2]]

        original_inference_count = self._inference_count

        self._inference_start_times.extend(start_times)
        self._inference_latencies.extend(latencies)

        for i, latency in enumerate(latencies):
            self._ov.include(latency)
            if latency > self._period:
                self._miss_events += 1
                self._missed_deadlines += int(latency / self._period)
                self._miss_inference_latencies.append(latency)
                self._miss_inference_start_times.append(start_times[i])
                self._avg_miss_latency = (
                    self._avg_miss_latency * (self._miss_events - 1) + latency
                ) / self._miss_events

        self._inference_count += len(latencies)
        self._miss_rate = self._miss_events / self._inference_count

        self._rolling_avg_latency = mean(latencies)
        self._avg_latency = self._avg_latency * (
            original_inference_count / self._inference_count
        ) + self._rolling_avg_latency * (
            len(latencies) / (self._inference_count)
        )

        max_latency = max(latencies)
        min_latency = min(latencies)
        if max_latency > self._max_latency:
            self._max_latency = max_latency
        if min_latency < self._min_latency:
            self._min_latency = min_latency

        self._stddev_latency = (
            self._ov.std if self._inference_count > 2 else 0.0
        )
        self._p99_latency = self._avg_latency + (2.326 * self._stddev_latency)

    def load_model(
        self,
    ):
        # Create a load request
        load_request = {
            "model_name": self._model_name,
            "load_type": self._load_type,
            "method": self._method,
            "batch_size": self._request_batch_size,
            "perf_targets": {
                "objectives": {
                    "perf_throughput": self._throughput_objective_weight,
                    "perf_latency": self._latency_objective_weight,
                },
                "constraints": {
                    "perf_throughput": self._throughput_constraint,
                    "perf_latency": self._latency_constraint,
                },
            },
        }

        url = f"http://{self._host}:{self._port}/load"
        res = requests.post(url, json=load_request)

        # Get the request uuid from the response if model was loaded succesfully
        # Also set the current model config dict based on response
        if res.status_code in [201, 303]:
            res_json = res.json()

            self._endpoint_uuid = res_json["request_uuid"]

            self._model_config_dict = res_json["model_config"]

        return res

    def upload_model(self, with_profile_data=False):
        """
        Upload a model and its triton model config to the AC
        Returns status code
        """
        url = f"http://{self._host}:{self._port}/upload/{self._model_type}"
        req_params = {"model_name": self._model_name}

        upload_files = [
            ("files", open(self._model_file, "rb")),
            (
                "files",
                text_format.MessageToBytes(
                    json_format.ParseDict(
                        self._model_config_dict, mc.ModelConfig()
                    )
                ),
            ),
        ]

        if with_profile_data:
            if self._platform_name:
                metrics_filename = f"{self._model_path}/metrics-model-inference-{self._platform_name}.csv"
            else:
                metrics_filename = (
                    f"{self._model_path}/metrics-model-inference.csv"
                )
            upload_files.append(
                (
                    "files",
                    open(metrics_filename, "rb"),
                )
            )

        res = requests.post(url, params=req_params, files=upload_files)
        return res
