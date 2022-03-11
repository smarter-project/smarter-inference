import logging
import os
import re
import select
import subprocess
import time
import uuid
from collections import defaultdict
from enum import Enum, auto
from fractions import Fraction
from threading import Thread
from time import sleep
from typing import Dict, List, Optional, Tuple

import psutil
import tritonclient.http as httpclient
from admission_controller.app.deadlines import Deadlines
from google.protobuf import json_format, text_format  # type: ignore
from model_analyzer.device.gpu_device import GPUDevice
from model_analyzer.model_analyzer_exceptions import (
    TritonModelAnalyzerException,
)
from model_analyzer.triton.model.model_config import ModelConfig
from model_analyzer.triton.server.server_config import TritonServerConfig
from model_analyzer.triton.server.server_factory import TritonServerFactory
from model_analyzer.triton.server.server_local import TritonServerLocal
from tritonclient.grpc import model_config_pb2

from .admission_controller_exceptions import *
from .config_generator.model_config_generator_factory import (
    ModelConfigGeneratorFactory,
)
from .constants import *
from .model import Metrics, ModelLoadRequest
from .model_state import ModelState

logger = logging.getLogger(__name__)
logger.propagate = False


class TritonManager:
    class Sched(Enum):
        DEFAULT = auto()
        NICE = auto()
        DEADLINE = auto()
        RR = auto()

    def __init__(self, sched_technique: Sched = Sched.DEFAULT) -> None:
        self._model_states: Dict[str, ModelState] = defaultdict(ModelState)
        self._model_threads: Dict[str, List[int]] = {}
        self._sched_technique = sched_technique
        self._hyperperiod = 0.0
        self._system_cpu_util_projected = 0.0
        self._max_cpu_util_projected = 0.0
        self._system_cpu_used_mem_projected = 0.0
        self._triton_base_mem_uss = 0.0
        self._gpu_util = 0.0
        self._re_gpu_util = re.compile(r"GR3D_FREQ (\d*(\.\d*)?)%")
        self._triton_base_threads: List[int] = []

    @property
    def model_states(self):
        return self._model_states

    @property
    def loaded_model_names(self):
        return self._model_states.keys()

    @property
    def hyperperiod(self):
        return self._hyperperiod

    @property
    def system_cpu_util_projected(self):
        return self._system_cpu_util_projected

    @property
    def max_cpu_util_projected(self):
        return self._max_cpu_util_projected

    @property
    def system_cpu_used_mem_projected(self):
        return self._system_cpu_used_mem_projected

    def get_triton_configs(self):
        return None

    def _wait_for_server_ready(self, num_retries):
        """
        Parameters
        ----------
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception
        Raises
        ------
        TritonModelAnalyzerException
            If server readiness could not be
            determined in given num_retries
        """

        retries = num_retries
        while retries > 0:
            try:
                if self._triton_client.is_server_ready():
                    time.sleep(1)
                    return
                else:
                    time.sleep(1)
                    retries -= 1
            except Exception as e:
                time.sleep(1)
                retries -= 1
                if retries == 0:
                    raise AdmissionControllerException(e)
        raise AdmissionControllerException(
            "Could not determine server readiness. Number of retries exceeded."
        )

    def _wait_for_model_ready(self, model_name, num_retries):
        """
        Returns when model is ready.

        Parameters
        ----------
        model_name : str
            name of the model to load from repository
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception

        Raises
        ------
        TritonModelAnalyzerException
            If could not determine model readiness
            in given num_retries
        """

        retries = num_retries
        while retries > 0:
            try:
                if self._triton_client.is_model_ready(model_name):
                    return
                else:
                    time.sleep(1)
                    retries -= 1
            except Exception as e:
                time.sleep(1)
                retries -= 1
                if retries == 0:
                    raise AdmissionControllerException(e)
        raise AdmissionControllerException(
            "Could not determine model readiness. Number of retries exceeded."
        )

    def healthy(self):
        return self._triton_client.is_server_live()

    def start_servers(self):
        self._triton_server = self._get_triton_server_handle()

        if SHIELD:
            self._start_triton_in_shield(self._triton_server)
        else:
            self._triton_server.start()

        # Set linux process information for server
        self._triton_server_psutil = psutil.Process(
            self._triton_server._tritonserver_process.pid
        )

        # Pin tritonserver process to list of triton_cpus
        if not (SHIELD):
            self._triton_server_psutil.cpu_affinity(AVAILABLE_TRITON_CPUS)

        self._triton_client = httpclient.InferenceServerClient(
            url=TRITON_URL,
            network_timeout=300.0,
            connection_timeout=300.0,
        )

        self._wait_for_server_ready(CLIENT_MAX_RETRIES)

        # Get triton base mem
        self._triton_base_mem_uss = (
            self._triton_server_psutil.memory_full_info().uss
        )

        # Get triton base threads
        self._triton_base_threads = [
            thread.id for thread in self._triton_server_psutil.threads()
        ]

        # Start model stats collection daemon in thread
        self._model_stats_daemon = Thread(
            target=self._update_model_inference_stats, daemon=True
        )
        self._model_stats_daemon.start()

        # Start GPU stat collection agent in thread
        if len(GPUS) > 0:
            self._gpu_stats_daemon = Thread(
                target=self._tegra_stats_manager, daemon=True
            )
            self._gpu_stats_daemon.start()

    def stop(self):
        self._triton_server.stop()

    def _start_triton_in_shield(
        self, triton_server_local: TritonServerLocal, env=None
    ):
        """
        Starts the tritonserver container locally in shield using
        cset shield
        """

        if triton_server_local._server_path:
            # Create command list and run subprocess
            cmd: List[str] = [
                "cset",
                "shield",
                "--exec",
                triton_server_local._server_path,
                "--",
            ]
            cmd += triton_server_local._server_config.to_args_list()

            # Set environment, update with user config env
            triton_env = os.environ.copy()

            if env:
                # Filter env variables that use env lookups
                for variable, value in env.items():
                    if value.find("$") == -1:
                        triton_env[variable] = value
                    else:
                        # Collect the ones that need lookups to give to the shell
                        triton_env[variable] = os.path.expandvars(value)

            # List GPUs to be used by tritonserver
            triton_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                [gpu.device_uuid() for gpu in triton_server_local._gpus]
            )

            if triton_server_local._log_path:
                try:
                    triton_server_local._log_file = open(
                        triton_server_local._log_path, "a+"
                    )
                except OSError as e:
                    raise AdmissionControllerException(e)
            else:
                triton_server_local._log_file = subprocess.DEVNULL

            # Construct Popen command
            try:
                triton_server_local._tritonserver_process = subprocess.Popen(
                    cmd,
                    stdout=triton_server_local._log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    universal_newlines=True,
                    env=triton_env,
                )

                logger.info("Triton Server started in shield.")
            except Exception as e:
                raise AdmissionControllerException(e)

    def _get_triton_server_handle(
        self,
        id: Optional[str] = None,
        start_port: int = 8000,
        http_thread_count: Optional[int] = None,
    ) -> TritonServerLocal:
        """
        Creates and returns a TritonServer
        """
        triton_config = TritonServerConfig()

        # A dirty hack here lets us add the missing rate-limit config option in
        # the Nvidia model analyzer source
        triton_config._server_args["rate-limit"] = None

        if id:
            triton_config["id"] = id
            log_path = f"/tmp/triton_log_{id}.txt"
        else:
            log_path = "/tmp/triton_log.txt"
        triton_config["model-repository"] = OUTPUT_REPO_PATH
        if http_thread_count:
            triton_config["http-thread-count"] = http_thread_count
        triton_config["http-port"] = str(start_port)
        triton_config["grpc-port"] = str(start_port + 1)
        triton_config["metrics-port"] = str(start_port + 2)
        triton_config["model-control-mode"] = "explicit"
        triton_config["rate-limit"] = "off"
        triton_config["log-verbose"] = "0"
        triton_config["backend-directory"] = os.path.join(
            TRITON_INSTALL_PATH, "backends"
        )
        server = TritonServerFactory.create_server_local(
            path=os.path.join(TRITON_INSTALL_PATH, "bin", "tritonserver"),
            config=triton_config,
            gpus=[
                GPUDevice(
                    "dummy_name", gpu_id, "dummy_pci_bus_id", str(gpu_id)
                )
                for gpu_id in GPUS
            ],
            log_path=log_path,
        )

        return server

    def _get_hyperperiod(self):
        """
        Generate the hyperperiod (lcm) of the set of model requests to be scheduled

        Returns:
            int: hyperperiod
        """
        periods = [
            Fraction(1 / x.min_throughput_constraint)
            for x in self._model_states.values()
            if x.min_throughput_constraint
        ]
        return Deadlines.get_hyperperiod(periods)

    def _parse_gpu_util(self, line: str):
        m = self._re_gpu_util.search(line)
        if m:
            return m.group(1)
        else:
            return 0.0

    def _tegra_stats_manager(self):
        self._tegra_stats_process = subprocess.Popen(
            ["/usr/bin/tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )

        try:
            ep = select.epoll()
            ep.register(
                self._tegra_stats_process.stdout.name,
                select.EPOLLIN | select.EPOLLHUP,
            )
            while not (self._tegra_stats_process.poll()):
                evts = ep.poll()
                for fd, evt in evts:
                    if evt & select.EPOLLIN:
                        line = (
                            self._tegra_stats_process.stdout.readline().rstrip()
                        )
                        self._gpu_util = self._parse_gpu_util(line)
                    if evt & select.EPOLLHUP:
                        for line in self._tegra_stats_process.stdout:
                            self._gpu_util = self._parse_gpu_util(
                                line.rstrip()
                            )
                        ep.unregister(fd)

        except Exception as e:
            logger.error(e)

        finally:
            ep.close()

    def _update_model_inference_stats(self):
        """
        Update the model metrics for all triton models managed in the system
        using the triton client api

        Raises:
            AdmissionControllerException: If failed to retrieve metrics
        """
        triton_client = httpclient.InferenceServerClient(
            url=TRITON_URL,
            network_timeout=300.0,
            connection_timeout=300.0,
        )
        sleep(5)

        while True:
            try:
                stats = triton_client.get_inference_statistics()
            except TimeoutError as e:
                sleep(1)
                continue
            except Exception as e:
                raise AdmissionControllerException(
                    "Failed to fetch model stats from tritonserver with"
                    f" exception {e}"
                )

            for model_stat in stats["model_stats"]:
                # Update model server inference data using averages since last call
                if (
                    len(self._model_states[model_stat["name"]].triton_stats)
                    > 0
                ):
                    new_inferences = (
                        model_stat["inference_stats"]["success"]["count"]
                        - self._model_states[model_stat["name"]].triton_stats[
                            "inference_stats"
                        ]["success"]["count"]
                    )
                    time_delta = (
                        model_stat["inference_stats"]["success"]["ns"]
                        - self._model_states[model_stat["name"]].triton_stats[
                            "inference_stats"
                        ]["success"]["ns"]
                    )
                    if new_inferences > 0:
                        model_stat["rolling_latency"] = (
                            time_delta / new_inferences
                        )
                        model_stat["slowdown"] = (
                            model_stat["rolling_latency"]
                            / 1e6
                            / self._model_states[
                                model_stat["name"]
                            ].metrics_projected.perf_latency_p99
                        )
                self._model_states[model_stat["name"]].triton_stats.update(
                    model_stat
                )

            sleep(1)

    def _add_load_request(
        self,
        uuid: str,
        load_request: ModelLoadRequest,
        max_batch_size: int = 1,
    ) -> Tuple[Optional[Dict], Optional[str], int]:

        # Update model state with new load request, will be unloaded later if
        # viable model configuration not found
        self._model_states[load_request.model_name].add_load_request(
            uuid, load_request, max_batch_size
        )

        # Update hyperperiod
        self._hyperperiod = self._get_hyperperiod()

        model_config_generator = (
            ModelConfigGeneratorFactory.create_model_config_generator(
                load_request,
                self.get_system_metrics_actual(),
                self._model_states,
                self._hyperperiod,
            )
        )

        try:
            (
                model_config_dict,
                metrics_projected,
                deadline_misses,
            ) = model_config_generator.process_triton_model_configuration()
        except AdmissionControllerException as e:
            logger.warning(f"Failed to load model with reason: {e}")
            self._model_states[load_request.model_name].remove_load_request(
                uuid
            )
            # Update hyperperiod
            self._hyperperiod = self._get_hyperperiod()
            return None, str(e), 0

        self._model_states[
            load_request.model_name
        ].model_config_dict = model_config_dict

        # Update model and system metrics
        if metrics_projected:
            self.update_model_metrics_projected(
                load_request.model_name, metrics_projected
            )
            if metrics_projected.system_cpu_util:
                self._system_cpu_util_projected = (
                    metrics_projected.system_cpu_util
                )

        self._max_cpu_util_projected = self._get_max_cpu_util_projected()
        self._system_cpu_used_mem_projected = (
            self._get_system_cpu_used_mem_projected()
        )

        return model_config_dict, None, deadline_misses

    def get_triton_metrics(self):
        metrics = {}
        if self._triton_server_psutil:
            total_mem = psutil.virtual_memory()[0]
            with self._triton_server_psutil.oneshot():
                metrics[
                    "triton_cpu_util"
                ] = self._triton_server_psutil.cpu_percent(interval=None) / (
                    NUM_TRITON_CPUS
                )
                metrics["projected_cpu_util"] = self._system_cpu_util_projected
                triton_mem_info = self._triton_server_psutil.memory_full_info()
                metrics["triton_memory_uss"] = triton_mem_info.uss
                metrics[
                    "projected_used_mem"
                ] = self.system_cpu_used_mem_projected
                metrics["projected_triton_memory_util"] = (
                    (self.system_cpu_used_mem_projected) / total_mem
                ) * 100
                metrics[
                    "triton_memory_util"
                ] = self._triton_server_psutil.memory_percent("uss")
                metrics[
                    "triton_threads"
                ] = self._triton_server_psutil.num_threads()

        return metrics

    def get_model_state(self, model_name: str):
        return self._model_states[model_name]

    def get_triton_model_stats(self) -> Dict:
        return {
            model_name: self._model_states[model_name].triton_stats
            for model_name in self._model_states
        }

    def get_model_config_dict(self, model_name: str) -> Dict:
        return self._model_states[model_name].model_config_dict

    def update_model_metrics_projected(
        self, model_name: str, metrics: Metrics
    ):
        self._model_states[model_name].metrics_projected = metrics

    def _load_triton_model(self, model_name):
        # Attempt to load or reload model
        self._triton_client.load_model(model_name=model_name)

        self._wait_for_model_ready(
            model_name=model_name,
            num_retries=CLIENT_MAX_RETRIES,
        )

        # Attempt to get thread ids by subtracting current thread set
        # from base thread set
        current_threads = [
            thread.id for thread in self._triton_server_psutil.threads()
        ]
        self._model_threads[model_name] = list(
            set(current_threads) - set(self._triton_base_threads)
        )
        logger.info(f"{model_name} threads: {self._model_threads[model_name]}")
        # Update base threads
        self._triton_base_threads = current_threads

        # If using nice, set nice levels accordingly
        # Depending on scheduling technique, modify the sched attributes
        # for each tritonserver based on model SLOs
        if PRIORITY_ASSIGNMENT == "SLACK":
            sorted_model_states = self._get_model_states_by_slack(
                descending=False
            )
        else:
            sorted_model_states = self._get_model_states_by_throughput(
                descending=False
            )
        if self._sched_technique == TritonManager.Sched.NICE:
            # Set tritonserver priorities
            nice = -1
            for model_name, _ in sorted_model_states:
                for thread_id in self._model_threads[model_name]:
                    psutil.Process(thread_id).nice(nice)
                nice -= 1

        elif self._sched_technique == TritonManager.Sched.RR:
            # Set tritonserver priorities
            rt_priority = 1
            for model_name, _ in sorted_model_states:
                for thread_id in self._model_threads[model_name]:
                    ret = self._set_rt_sched(
                        pid=thread_id, priority=rt_priority
                    )
                    if ret:
                        raise AdmissionControllerException(
                            f"Failed to set SCHED_RR for {model_name}"
                        )
                rt_priority += 1

        elif self._sched_technique == TritonManager.Sched.DEADLINE:
            runtime = self._model_states[
                model_name
            ].metrics_projected.perf_latency_p99
            if self._model_states[model_name].min_throughput_constraint:
                deadline = (
                    1
                    / self._model_states[model_name].min_throughput_constraint
                ) * 1e3
            elif self._model_states[model_name].max_latency_constraint:
                deadline = self._model_states[
                    model_name
                ].max_latency_constraint
            else:
                raise AdmissionControllerException(
                    f"Model state: {self._model_states[model_name]} does not"
                    " have throughput or latency constraint"
                )
            # Set SCHED_DEADLINE for each inference thread of model
            for thread_id in self._model_threads[model_name]:
                ret = self._set_rt_sched(
                    pid=thread_id, runtime=int(runtime), deadline=int(deadline)
                )
                if ret:
                    raise AdmissionControllerException(
                        f"Failed to set SCHED_DEADLINE for {model_name}"
                    )

    def _set_rt_sched(
        self,
        pid: int,
        priority: int = 0,
        runtime: Optional[int] = None,
        deadline: Optional[int] = None,
        period: Optional[int] = None,
    ):
        """
        Use chrt to set SCHED_DEADLINE or SCHED_RR to inference thread based
        on PIDs

        Args:
            pid (int): pid of task of interest
            priority (int): priority of rt schedule class
            runtime (Optional[int]): cpu runtime target of threads in ms
            deadline (Optional[int]): deadline target of threads in ms
            period (Optional[int]): period of threads in ms
        """
        if self._sched_technique == TritonManager.Sched.DEADLINE:
            if not (runtime) or not (deadline):
                raise AdmissionControllerException(
                    "DEADLINE class used but runtime or deadline not passed"
                )
            if not (period):
                period = deadline

            res = subprocess.run(
                [
                    "chrt",
                    "-d",
                    "-T",
                    str(runtime * int(1e6)),
                    "-D",
                    str(deadline * int(1e6)),
                    "-P",
                    str(period * int(1e6)),
                    "--pid",
                    str(priority),
                    str(pid),
                ],
                capture_output=True,
            )
        else:
            if priority == 0:
                priority = 1
            res = subprocess.run(
                [
                    "chrt",
                    "-r",
                    "--pid",
                    str(priority),
                    str(pid),
                ],
                capture_output=True,
            )

        if res.stdout:
            logger.info(res.stdout)
        if res.stderr:
            logger.error(res.stderr)

        return res.returncode

    def load_model(self, load_request: ModelLoadRequest):
        model_config_dict = None

        # Generate UUID for this load request
        load_request_uuid = str(uuid.uuid1())

        model_dir = os.path.join(OUTPUT_REPO_PATH, load_request.model_name)

        # Before we load the model we must determine if the model will fit, and if so,
        # the best configuration given the current system state.
        # ModelConfigGenerator within the TritonManager handles this.
        # We first update model state with the proposal for the new load request, then invoke
        # the model loader to generate a triton configuration and get updated projected metrics
        max_batch_size = int(
            ModelConfig.create_from_file(model_dir).get_config()[
                "max_batch_size"
            ]
        )

        (model_config_dict, reason, deadline_misses,) = self._add_load_request(
            load_request_uuid,
            load_request,
            max_batch_size=max_batch_size,
        )

        # If load request unsatisfiable, raise exception
        if not (model_config_dict):
            raise UnsatisfiableRequestException(
                f"Unsatisfiable performance objectives. Reason {reason}"
            )

        # Write model configuration to the repository
        self._write_model_config(
            model_dir, load_request.model_name, model_config_dict
        )

        # At this point we are commited to loading the model in triton
        self._load_triton_model(load_request.model_name)

        return {
            "model_config": model_config_dict,
            "request_uuid": load_request_uuid,
            "deadline_misses": deadline_misses,
        }

    def unload_model(self, model_name: str):
        self._wait_for_server_ready(CLIENT_MAX_RETRIES)

        self._triton_client.unload_model(model_name=model_name)

        # Update model states
        if model_name in self.loaded_model_names:
            self._delete_model(model_name)

    def _delete_model(self, model_name: str):
        del self._model_states[model_name]

    def _write_model_config(
        self, model_dir: str, model_name: str, model_config_dict: Dict
    ):
        """
        Write model config to config.pbtxt file in model repo
        """
        self._model_states[model_name].model_config_dict = model_config_dict

        model_config_file_path = os.path.join(model_dir, "config.pbtxt")
        protobuf_message = json_format.ParseDict(
            model_config_dict, model_config_pb2.ModelConfig()
        )
        model_config_bytes = text_format.MessageToBytes(protobuf_message)
        with open(model_config_file_path, "wb") as f:
            f.write(model_config_bytes)

    # System level projected cpu utilization is based on the sum of
    # the constituent model max projected cpu utilizations
    def _get_max_cpu_util_projected(self, model_list=None) -> float:
        if model_list:
            model_names = model_list
        else:
            model_names = self.loaded_model_names
        return sum(
            [
                self._model_states[name].metrics_projected.max_cpu_util
                for name in model_names
                if self._model_states[name].metrics_projected.max_cpu_util
            ]
        )

    # System level projected cpu utilization is based on the sums of
    # the constituent model projected cpu utilizations
    def _get_system_cpu_used_mem_projected(self) -> float:
        total_model_mem_uss = sum(
            [
                self._model_states[name].metrics_projected.cpu_used_memory
                * 1e6
                - self._triton_base_mem_uss
                for name in self._model_states.keys()
                if self._model_states[name].metrics_projected.cpu_used_memory
            ]
        )
        return total_model_mem_uss + self._triton_base_mem_uss

    # System level projected cpu utilization is based on the sums of
    # the constituent model projected cpu utilizations
    def model_inference_metrics_projected(self) -> Dict[str, Dict]:
        return {
            name: self._model_states[name].metrics_projected.dict()
            for name in self._model_states.keys()
        }

    def get_system_metrics_actual(self) -> Metrics:
        cpu_util = psutil.cpu_percent(interval=None)
        cpu_times_percent = psutil.cpu_times_percent(interval=None)

        mem_stats = psutil.virtual_memory()
        used_mem = float(mem_stats[3])
        avail_mem = float(mem_stats[1])

        return Metrics(
            cpu_util=cpu_util,
            user_cpu_util=cpu_times_percent.user,
            system_cpu_util=cpu_times_percent.system,
            idle_cpu_util=cpu_times_percent.idle,
            iowait_cpu_util=cpu_times_percent.iowait,
            irq_cpu_util=cpu_times_percent.irq,
            softirq_cpu_util=cpu_times_percent.softirq,
            cpu_used_memory=used_mem,
            cpu_free_memory=avail_mem,
            gpu_util=self._gpu_util,
        )

    def model_loaded_same_config(self, load_request: ModelLoadRequest):
        try:
            return (
                load_request
                in self._model_states[
                    load_request.model_name
                ].load_requests.values()
            )
        except:
            return False

    def _get_model_states_by_throughput(
        self, descending: bool = True
    ) -> List[Tuple[str, ModelState]]:
        # Get list of model state items sorted by target throughput descending
        # or ascending
        return [
            v
            for v in sorted(
                self._model_states.items(),
                key=lambda item: item[1].min_throughput_constraint,
                reverse=descending,
            )
        ]

    def _get_model_states_by_slack(
        self, descending: bool = True
    ) -> List[Tuple[str, ModelState]]:
        # Get list of model state items sorted by slack in deadlines descending
        # or ascending
        return [
            v
            for v in sorted(
                self._model_states.items(),
                key=lambda item: (
                    (1 / item[1].min_throughput_constraint)
                    - item[1].metrics_projected.perf_latency_p99
                ),
                reverse=descending,
            )
        ]
