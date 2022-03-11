import logging
import time
from collections import defaultdict
from threading import Lock, Thread
from time import sleep
from typing import Dict

import psutil
import tritonclient.http as httpclient
from matplotlib.style import available
from model_analyzer.triton.server.server_local import TritonServerLocal
from tritonhttpclient import InferenceServerClient

from .admission_controller_exceptions import *
from .constants import *
from .triton_manager import TritonManager

logger = logging.getLogger(__name__)
logger.propagate = False


class MultiTritonManager(TritonManager):
    def get_triton_configs(self):
        return {
            model_name: server._server_config
            for model_name, server in self._triton_servers.items()
        }

    def _triton_port_generator(self, start_port: int = 8000):
        while True:
            yield start_port
            start_port += 3

    def _wait_for_model_server_ready(self, num_retries, model_name: str):
        """
        Parameters
        ----------
        num_retries : int
            number of times to send a ready status
            request to the server before raising
            an exception
        Raises
        ------
        AdmissionControllerException
            If server readiness could not be
            determined in given num_retries
        """

        retries = num_retries
        while retries > 0:
            try:
                self._triton_client_lock.acquire()
                ready = self._triton_clients[model_name].is_server_ready()
                if ready:
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
            finally:
                self._triton_client_lock.release()
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
        AdmissionControllerException
            If could not determine model readiness
            in given num_retries
        """

        retries = num_retries
        while retries > 0:
            try:
                self._triton_client_lock.acquire()
                ready = self._triton_clients[model_name].is_model_ready(
                    model_name
                )
                if ready:
                    return
                else:
                    time.sleep(1)
                    retries -= 1
            except Exception as e:
                time.sleep(1)
                retries -= 1
                if retries == 0:
                    raise AdmissionControllerException(e)
            finally:
                self._triton_client_lock.release()
        raise AdmissionControllerException(
            "Could not determine model readiness. Number of retries exceeded."
        )

    def healthy(self):
        healthy = True
        for triton_client in self._triton_clients.values():
            healthy = healthy and triton_client.is_server_live()
        return healthy

    def start_servers(self):
        # Initialize dicts which hold triton management tools
        # for each individual model
        self._triton_servers: Dict[str, TritonServerLocal] = {}
        self._triton_port_starts: Dict[str, int] = {}
        self._triton_clients: Dict[str, InferenceServerClient] = {}
        self._triton_server_psutils: Dict[str, psutil.Process] = {}

        self._triton_client_lock = Lock()

        self._port_generator = self._triton_port_generator()

        # Start model stats collection daemon in thread
        self._model_stats_daemon = Thread(
            target=self._update_model_inference_stats, daemon=True
        )
        self._model_stats_daemon.start()

        # Only start collecting gpu data, as triton server isn't started until
        # first model is loaded
        if len(GPUS) > 0:
            self._gpu_stats_daemon = Thread(
                target=self._tegra_stats_manager, daemon=True
            )
            self._gpu_stats_daemon.start()

    def stop(self):
        for server in self._triton_servers.values():
            server.stop()

    def _update_model_inference_stats(self):
        """
        Update the model metrics for all triton models managed in the system
        using the triton client api

        Raises:
            AdmissionControllerException: If failed to retrieve metrics
        """
        sleep(5)

        while True:
            for model_name in self._triton_clients:
                try:
                    self._triton_client_lock.acquire()
                    stats = self._triton_clients[
                        model_name
                    ].get_inference_statistics()
                except (TimeoutError, ConnectionRefusedError) as e:
                    sleep(1)
                    continue
                except Exception as e:
                    raise AdmissionControllerException(
                        "Failed to fetch model stats from tritonserver for"
                        f" {model_name} with exception {type(e)} {e}"
                    )
                finally:
                    self._triton_client_lock.release()

                for model_stat in stats["model_stats"]:
                    # Update model server inference data using averages since last call
                    if (
                        len(
                            self._model_states[model_stat["name"]].triton_stats
                        )
                        > 0
                    ):
                        new_inferences = (
                            model_stat["inference_stats"]["success"]["count"]
                            - self._model_states[
                                model_stat["name"]
                            ].triton_stats["inference_stats"]["success"][
                                "count"
                            ]
                        )
                        time_delta = (
                            model_stat["inference_stats"]["success"]["ns"]
                            - self._model_states[
                                model_stat["name"]
                            ].triton_stats["inference_stats"]["success"]["ns"]
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

    def get_triton_metrics(self):
        # Return aggregate information for all managed servers in the
        # system
        metrics = defaultdict(lambda: 0.0)
        total_mem = psutil.virtual_memory().total
        for model_name in self._triton_clients:
            if self._triton_server_psutils[model_name]:
                try:
                    with self._triton_server_psutils[model_name].oneshot():
                        metrics["triton_cpu_util"] += (
                            self._triton_server_psutils[
                                model_name
                            ].cpu_percent(interval=None)
                            / NUM_TRITON_CPUS
                        )
                        triton_mem_info = self._triton_server_psutils[
                            model_name
                        ].memory_full_info()
                        metrics["triton_memory_uss"] += triton_mem_info.uss
                        metrics[
                            "triton_memory_util"
                        ] += self._triton_server_psutils[
                            model_name
                        ].memory_percent(
                            "pss"
                        )
                        metrics[
                            "triton_threads"
                        ] += self._triton_server_psutils[
                            model_name
                        ].num_threads()
                except:
                    continue

        if len(self._model_states) > 1:
            metrics["triton_memory_uss"] += self._triton_base_mem_uss
        metrics["projected_used_mem"] = self._system_cpu_used_mem_projected
        metrics["projected_triton_memory_util"] = (
            (self._system_cpu_used_mem_projected) / total_mem
        ) * 100
        metrics["projected_cpu_util"] = self._system_cpu_util_projected

        return dict(metrics)

    def _load_triton_model(self, model_name):
        port_start = next(self._port_generator)

        # Start new triton server
        self._triton_port_starts[model_name] = port_start
        self._triton_servers[model_name] = self._get_triton_server_handle(
            id=model_name, start_port=port_start
        )
        if SHIELD:
            self._start_triton_in_shield(self._triton_servers[model_name])
        else:
            self._triton_servers[model_name].start()

        # Set linux process information for server
        self._triton_server_psutils[model_name] = psutil.Process(
            self._triton_servers[model_name]._tritonserver_process.pid
        )

        # Pin tritonserver process to list of triton cpus
        if not (SHIELD):
            self._triton_server_psutils[model_name].cpu_affinity(
                AVAILABLE_TRITON_CPUS
            )

        self._triton_clients[model_name] = httpclient.InferenceServerClient(
            url=f"localhost:{port_start}",
            network_timeout=300.0,
            connection_timeout=300.0,
        )

        self._wait_for_model_server_ready(CLIENT_MAX_RETRIES, model_name)

        # Get base thread ids for new server
        base_threads = [
            thread.id
            for thread in self._triton_server_psutils[model_name].threads()
        ]

        # Get triton base uss mem if only model loaded
        if len(self.loaded_model_names) == 1:
            mem_info = self._triton_server_psutils[
                model_name
            ].memory_full_info()
            self._triton_base_mem_uss = mem_info.uss
            self._triton_base_mem_pss = mem_info.pss

        self._triton_client_lock.acquire()
        self._triton_clients[model_name].load_model(model_name=model_name)
        self._triton_client_lock.release()

        self._wait_for_model_ready(
            model_name=model_name,
            num_retries=CLIENT_MAX_RETRIES,
        )

        # Attempt to get thread ids by subtracting current thread set
        # from base thread set
        current_threads = [
            thread.id
            for thread in self._triton_server_psutils[model_name].threads()
        ]
        self._model_threads[model_name] = list(
            set(current_threads) - set(base_threads)
        )
        logger.info(f"{model_name} threads: {self._model_threads[model_name]}")

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

        if self._sched_technique == TritonManager.Sched.RR:
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

    def unload_model(self, model_name: str):
        self._wait_for_model_server_ready(CLIENT_MAX_RETRIES, model_name)

        self._triton_client_lock.acquire()
        self._triton_clients[model_name].unload_model(model_name=model_name)
        self._triton_client_lock.release()

        # Update model states
        if model_name in self.loaded_model_names:
            self._delete_model(model_name)

    # System level projected mem utilization is based on the sums of
    # the constituent model projected mem utilizations
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
        logging.info(f"Total model mem uss: {total_model_mem_uss}")

        # Return sum of all the memory allocated for triton model uss'
        # with triton unique memory added back in
        return total_model_mem_uss + self._triton_base_mem_uss
