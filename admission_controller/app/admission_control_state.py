# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import pathlib
import shutil
import time
from typing import Dict

import psutil
import requests  # type: ignore
from admission_controller.app.admission_controller_exceptions import \
    UnsatisfiableRequestException
from fastapi import File, HTTPException, UploadFile, status
from model_analyzer.model_analyzer_exceptions import \
    TritonModelAnalyzerException
from tritonclient.utils import InferenceServerException

from ..nginx.server import NginxServer
from ..nginx.server_config import NginxServerConfig
from ..nginx.server_factory import NginxServerFactory
from .constants import *
from .model import *
from .multi_triton_manager import MultiTritonManager
from .triton_manager import TritonManager


class AdmissionControlState:
    def __init__(self):
        # Get sched technique from constants
        if SCHED == "NICE":
            sched_technique = TritonManager.Sched.NICE
        elif SCHED == "DEADLINE":
            sched_technique = TritonManager.Sched.DEADLINE
        elif SCHED == "RR":
            sched_technique = TritonManager.Sched.RR
        else:
            sched_technique = TritonManager.Sched.DEFAULT
        if MULTI_SERVER:
            self._triton_manager: TritonManager = MultiTritonManager(
                sched_technique=sched_technique
            )
        else:
            self._triton_manager: TritonManager = TritonManager(
                sched_technique=sched_technique
            )

        self._nginx_server = self._get_nginx_server_handle()

    def start_servers(self):
        # Start triton server(s)
        self._triton_manager.start_servers()

        # Handle nginx cpu affinity by setting it to use all cores besides
        # those assigned to triton to avoid interference
        self._nginx_server.start()
        nginx_server_psutil = psutil.Process(
            self._nginx_server._nginx_process.pid
        )

        total_cpus = list(range(0, multiprocessing.cpu_count()))
        # If triton is using all cpus, also allow nginx and api server to use all cpus
        if AVAILABLE_TRITON_CPUS == total_cpus:
            other_cpus = total_cpus
        else:
            other_cpus = list(set(total_cpus) - set(AVAILABLE_TRITON_CPUS))
        if not (SHIELD):
            nginx_server_psutil.cpu_affinity(other_cpus)

        # Also set affinity and nice for all nginx worker processes
        children = nginx_server_psutil.children(recursive=True)
        for child in children:
            print("Child pid is {}".format(child.pid))
            if not (SHIELD):
                psutil.Process(child.pid).cpu_affinity(other_cpus)

        # Also set cpu affinity of api server
        if not (SHIELD):
            psutil.Process(os.getpid()).cpu_affinity(other_cpus)

    def stop_servers(self):
        self._triton_manager.stop()
        self._nginx_server.stop()

    def health(self):
        """
        Verify both Triton and Nginx are running

        Returns:
            bool: System healthy or not
        """
        return self._triton_manager.healthy() and (
            requests.get("http://localhost:2521/health").status_code == 200
        )

    def _get_nginx_server_handle(self) -> NginxServer:
        """
        Creates and returns a NginxServer
        """

        nginx_config = NginxServerConfig(
            config_path=NGINX_CONFIG_PATH,
            nginx_logging=NGINX_LOGGING,
        )
        server = NginxServerFactory.create_server_local(
            path="nginx",
            config=nginx_config,
            log_path="",
        )
        server.update_config(self._triton_manager.model_states)

        return server

    def system_metrics(self) -> Dict:
        """
        Returns:
            Dict: dict containing all metrics relevant to admission controller
        """
        system_metrics = self._triton_manager.get_system_metrics_actual()
        load1, load5, load15 = psutil.getloadavg()

        metrics = {
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "model_inference_stats": self._triton_manager.get_triton_model_stats(),
            "model_inference_stats_projected": self._triton_manager.model_inference_metrics_projected(),
            "free_memory": system_metrics.cpu_free_memory,
            "system_wide_cpu_util": system_metrics.cpu_util,
            "gpu_util": system_metrics.gpu_util,
            "user_cpu_util": system_metrics.user_cpu_util,
            "system_cpu_util": system_metrics.system_cpu_util,
            "idle_cpu_util": system_metrics.idle_cpu_util,
            "iowait_cpu_util": system_metrics.iowait_cpu_util,
            "irq_cpu_util": system_metrics.irq_cpu_util,
            "softirq_cpu_util": system_metrics.softirq_cpu_util,
            "gpu_util": system_metrics.gpu_util,
            "time_stamp": time.time(),
        }

        return {**metrics, **self._triton_manager.get_triton_metrics()}

    def register_model(
        self,
        model_name: str,
        model_type: ModelType,
        files: List[UploadFile] = File(...),
        version: str = "1",
    ):
        """
        User uploads a model and it's corresponding triton model configuration.
        Optionally user can also upload the offline profile data generated by model analyzer.

        Model is created in model repo path specified in model analyzer config.
        """

        if len(files) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Please upload at least 2 files. Model file and model"
                    " config"
                ),
            )

        model_file = files[0]
        model_config = files[1]

        new_model_dir = os.path.join(OUTPUT_REPO_PATH, model_name)
        try:
            # Create the directory for the new model
            os.makedirs(new_model_dir)
        except FileExistsError:
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                detail="Model already exists with same name",
            )

        # Write the model file to the newly created directory
        if model_type == ModelType.tflite:
            model_filename = "model.tflite"
        elif model_type == ModelType.tf:
            model_filename = "model.graphdef"

        model_file_path = os.path.join(new_model_dir, version, model_filename)
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)

        # Write the user supplied model configuration to the directory
        model_config_file_path = os.path.join(new_model_dir, "config.pbtxt")
        with open(model_config_file_path, "wb") as buffer:
            shutil.copyfileobj(model_config.file, buffer)

        if len(files) <= 4:
            remaining_files = files[2:]
            for file in remaining_files:
                extension = pathlib.Path(file.filename).suffix
                if extension == ".csv":
                    # Write the user supplied profiling data to the directory
                    perf_csv_file_path = os.path.join(
                        new_model_dir, "perf.csv"
                    )
                    with open(perf_csv_file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                elif extension == ".classes":
                    classes_filepath = os.path.join(
                        new_model_dir, file.filename
                    )
                    with open(classes_filepath, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many files uploaded in request. Max 4",
            )

        return {"Upload": model_name}

    def load_model(self, load_request: ModelLoadRequest):
        try:
            response = self._triton_manager.load_model(load_request)
        except TritonModelAnalyzerException as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model load failed with exception {e}",
            )
        except InferenceServerException as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model failed to load",
            )
        except UnsatisfiableRequestException as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Unsatisfiable performance objectives. {e}",
            )

        # Update Nginx config with ratelimit data
        self._nginx_server.update_config(
            self._triton_manager.model_states,
            triton_configs=self._triton_manager.get_triton_configs(),
        )

        return response

    def unload_model(self, model_name: str):
        try:
            self._triton_manager.unload_model(model_name)
        except TritonModelAnalyzerException as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model unload failed with exception {e}",
            )
        except InferenceServerException as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model failed to unload",
            )

        # Update Nginx config with ratelimit data
        self._nginx_server.update_config(self._triton_manager.model_states)
