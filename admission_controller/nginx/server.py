from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..app.model_state import ModelState


class NginxServer(ABC):
    """
    Defines the interface for the objects created by
    TritonServerFactory
    """

    @abstractmethod
    def start(self):
        """
        Starts the tritonserver
        """

    @abstractmethod
    def stop(self):
        """
        Stops and cleans up after the server
        """

    @abstractmethod
    def cpu_stats(self):
        """
        Returns the CPU memory usage and CPU available memory in MB
        """

    @abstractmethod
    def update_config(
        self,
        model_states: Dict[str, ModelState],
        triton_configs: Optional[List[Dict]] = None,
    ):
        """
        Update the nginx config file

        Parameters
        ----------
        model_states: dict
            keys are argument names and values are their values.
        """
