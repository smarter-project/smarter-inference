from typing import Dict

from numpy import product

from .admission_controller_exceptions import AdmissionControllerException
from .model import *


class ModelState:
    def __init__(self):
        # These will be kept track of model wide
        # (ie handle multiple applications requesting the same model)
        self._load_requests: Dict[str, ModelLoadRequest] = {}
        self._max_batch_size: int = 1
        self._model_config_dict: Dict = {}
        self._metrics_actual = Metrics()
        self._metrics_projected = Metrics()
        self._triton_stats: Dict = {}

    @property
    def load_requests(self):
        return self._load_requests

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, max_batch_size: int):
        self._max_batch_size = max_batch_size

    @property
    def model_config_dict(self):
        return self._model_config_dict

    @model_config_dict.setter
    def model_config_dict(self, model_config_dict: Dict):
        self._model_config_dict = model_config_dict
        self._max_input_size = self._get_max_input_size()

    @property
    def max_input_size(self):
        return self._max_input_size

    @property
    def metrics_actual(self):
        return self._metrics_actual

    @metrics_actual.setter
    def metrics_actual(self, metrics_actual: Metrics):
        self._metrics_actual = metrics_actual

    @property
    def metrics_projected(self):
        return self._metrics_projected

    @metrics_projected.setter
    def metrics_projected(self, metrics_projected: Metrics):
        self._metrics_projected = metrics_projected

    @property
    def triton_stats(self):
        return self._triton_stats

    @triton_stats.setter
    def triton_stats(self, triton_stats: Dict):
        self._triton_stats = triton_stats

    @property
    def total_requested_batch_size(self):
        # Return sum of all the batch sizes from the load requests
        return sum(
            [
                x.batch_size
                for x in self._load_requests.values()
                if x.batch_size
            ]
        )

    @property
    def max_requested_batch_size(self):
        # Return max of all the batch sizes from the load requests
        return max(
            [
                x.batch_size
                for x in self._load_requests.values()
                if x.batch_size
            ]
        )

    @property
    def concurrent_clients(self):
        return len(self._load_requests)

    @property
    def max_latency_constraint(self):
        # Return the lowest latency of all the load requests
        return self._sort_metric(
            [
                x.perf_targets.constraints.perf_latency_p99
                for x in self._load_requests.values()
                if x.perf_targets.constraints.perf_latency_p99
            ]
        )

    @property
    def min_throughput_constraint(self) -> float:
        # Return the sum of all the throughputs for load requests
        return self._sum_metric(
            [
                x.perf_targets.constraints.perf_throughput
                for x in self._load_requests.values()
                if x.perf_targets.constraints
            ]
        )

    @property
    def min_gpu_used_memory_constraint(self):
        return self._sum_metric(
            [
                x.perf_targets.constraints.gpu_used_memory
                for x in self._load_requests.values()
                if x.perf_targets.constraints.gpu_used_memory
            ]
        )

    @property
    def max_gpu_free_memory_constraint(self):
        return self._sort_metric(
            [
                x.perf_targets.constraints.gpu_free_memory
                for x in self._load_requests.values()
                if x.perf_targets.constraints.gpu_free_memory
            ]
        )

    @property
    def min_cpu_util_constraint(self):
        return self._sum_metric(
            [
                x.perf_targets.constraints.cpu_util
                for x in self._load_requests.values()
                if x.perf_targets.constraints.cpu_util
            ]
        )

    @property
    def min_gpu_utilization_constraint(self):
        return self._sum_metric(
            [
                x.perf_targets.constraints.gpu_utilization
                for x in self._load_requests.values()
                if x.perf_targets.constraints.gpu_utilization
            ]
        )

    @property
    def min_cpu_used_memory_constraint(self):
        return self._sum_metric(
            [
                x.perf_targets.constraints.cpu_used_memory
                for x in self._load_requests.values()
                if x.perf_targets.constraints.cpu_used_memory
            ]
        )

    @property
    def max_cpu_free_ram_constraint(self):
        return self._sort_metric(
            [
                x.perf_targets.constraints.cpu_free_ram
                for x in self._load_requests.values()
                if x.perf_targets.constraints.cpu_free_ram
            ]
        )

    @property
    def perf_constraints(self):
        perf_constraints = Metrics(
            perf_throughput=self.min_throughput_constraint,
            perf_latency_p99=self.max_latency_constraint,
            gpu_used_memory=self.min_gpu_used_memory_constraint,
            cpu_util=self.min_cpu_util_constraint,
            cpu_used_memory=self.min_cpu_used_memory_constraint,
        )

        if all(x == None for x in perf_constraints.dict().values()):
            return None
        else:
            return perf_constraints

    def add_load_request(
        self, uuid: str, load_request: ModelLoadRequest, max_batch_size: int
    ):
        # Add load request to map
        self._load_requests[uuid] = load_request
        self._max_batch_size = max_batch_size

    def remove_load_request(self, uuid: str):
        # Remove this load request from the map
        del self._load_requests[uuid]

    def _sum_metric(self, metric_list):
        if len(metric_list) > 0:
            return sum(metric_list)
        else:
            return None

    def _sort_metric(self, metric_list, reverse=False):
        if len(metric_list) > 0:
            if reverse:
                return max(metric_list)
            else:
                return min(metric_list)
        else:
            return None

    def _get_max_input_size(self):
        input_size = 0
        for input in self._model_config_dict["input"]:
            input_size += product(
                [int(x) for x in input["dims"]]
            ) * self._input_type_to_bytes(input["data_type"])

        return input_size * self.max_batch_size

    def _input_type_to_bytes(self, input_type: str) -> int:
        if input_type in ["TYPE_FP64", "TYPE_INT64", "TYPE_UINT64"]:
            return 8
        elif input_type in ["TYPE_FP32", "TYPE_INT32", "TYPE_UINT32"]:
            return 4
        elif input_type in [
            "TYPE_FP16",
            "TYPE_INT16",
            "TYPE_UINT16",
            "TYPE_BF16",
        ]:
            return 2
        elif input_type in ["TYPE_INT8", "TYPE_UINT8"]:
            return 1
        else:
            raise AdmissionControllerException(
                f"Type {input_type} Unknown or Unsupported for model input"
            )
