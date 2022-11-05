# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import signal
from subprocess import DEVNULL, STDOUT, Popen, TimeoutExpired
from typing import Dict, List, Optional

import psutil
from model_analyzer.constants import LOGGER_NAME, SERVER_OUTPUT_TIMEOUT_SECS
from model_analyzer.model_analyzer_exceptions import \
    TritonModelAnalyzerException

from ..app.model_state import ModelState
from .server import NginxServer
from .server_config import NginxServerConfig

logger = logging.getLogger(__name__)
logger.propagate = False


class NginxServerLocal(NginxServer):
    """
    Concrete Implementation of NginxServer interface that runs
    nginx server locally as as subprocess.
    """

    def __init__(self, path, config, log_path):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the nginx server executable
        config : NginxServerConfig
            the config object containing arguments for this server instance
        log_path: str
            Absolute path to the triton log file
        """

        self._nginx_process = None
        self._server_path: str = path
        self._nginx_config: NginxServerConfig = config
        self._log_path: str = log_path

    def start(self):
        """
        Starts the nginx server process locally
        """

        if self._server_path:
            # Create nginx config and run subprocess
            cmd = [
                self._server_path,
                "-c",
                self._nginx_config.get_config_path(),
            ]

            # Write result of jinja template render to nginx config file
            self._write_config()

            if self._log_path:
                try:
                    self._log_file = open(self._log_path, "a+")
                except OSError as e:
                    raise TritonModelAnalyzerException(e)
            else:
                self._log_file = DEVNULL
            self._nginx_process = Popen(
                cmd,
                stdout=self._log_file,
                stderr=STDOUT,
                start_new_session=True,
                universal_newlines=True,
            )

            logger.info("Nginx Server started.")

    def _write_config(self):
        with open(self._nginx_config.get_config_path(), "w+") as output_file_:
            output_file_.write(self._nginx_config.get_config())

    def stop(self):
        """
        Stops the running nginx server
        """

        # Terminate process, capture output
        if self._nginx_process is not None:
            self._nginx_process.terminate()
            try:
                self._nginx_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS
                )
            except TimeoutExpired:
                self._nginx_process.kill()
                self._nginx_process.communicate()
            self._nginx_process = None
            if self._log_path:
                self._log_file.close()
            logger.info("Nginx Server stopped.")

    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

        if self._nginx_process:
            server_process = psutil.Process(self._nginx_process.pid)
            process_memory_info = server_process.memory_full_info()
            system_memory_info = psutil.virtual_memory()

            # Divide by 1.0e6 to convert from bytes to MB
            return (process_memory_info.uss // 1.0e6), (
                system_memory_info.available // 1.0e6
            )
        else:
            return 0.0, 0.0

    def update_config(
        self,
        model_states: Dict[str, ModelState],
        triton_configs: Optional[List[Dict]] = None,
    ):
        """
        Update the nginx config file and rewrite it to file.
        Reloads the nginx instance to account for the new configuration

        Parameters
        ----------
        model_states: dict
            keys are model names and values are load requests.
        """

        self._nginx_config.model_states = model_states
        self._nginx_config.triton_configs = triton_configs

        # This overwrites the existing nginx config file
        self._nginx_config.update()
        self._write_config()

        # Sending sighup to the nginx process causes it to reload it's configuration without dying
        if self._nginx_process is not None:
            self._nginx_process.send_signal(signal.SIGHUP)
