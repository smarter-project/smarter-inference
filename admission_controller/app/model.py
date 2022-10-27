from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ModelType(str, Enum):
    tflite = "tflite"
    tf = "tf"


class LoadType(str, Enum):
    existing = "existing"
    auto_gen = "auto_gen"


class ConfigGeneration(str, Enum):
    lookup = "lookup"
    lookup_lr = "lookup_lr"
    infer_gcn = "infer_gcn"
    infer_vw = "infer_vw"
    random = "random"
    passthrough = "passthrough"


class Metrics(BaseModel):
    cpu_util: Optional[float] = None
    user_cpu_util: Optional[float] = None
    system_cpu_util: Optional[float] = None
    idle_cpu_util: Optional[float] = None
    iowait_cpu_util: Optional[float] = None
    irq_cpu_util: Optional[float] = None
    softirq_cpu_util: Optional[float] = None
    max_cpu_util: Optional[float] = None
    cpu_used_memory: Optional[float] = None
    cpu_free_memory: Optional[float] = None
    gpu_used_memory: Optional[float] = None
    gpu_free_memory: Optional[float] = None
    gpu_util: Optional[float] = None
    perf_throughput: Optional[float] = None
    perf_latency_p99: Optional[int] = None

    def constraints_met(self, target: Metrics) -> bool:
        if self.perf_throughput and target.perf_throughput:
            if self.perf_throughput < target.perf_throughput:
                return False

        if self.perf_latency_p99 and target.perf_latency_p99:
            if self.perf_latency_p99 > target.perf_latency_p99:
                return False

        if self.cpu_used_memory and target.cpu_used_memory:
            if self.cpu_used_memory > target.cpu_used_memory:
                return False

        if self.cpu_free_memory and target.cpu_free_memory:
            if self.cpu_free_memory < target.cpu_free_memory:
                return False

        if self.cpu_util and target.cpu_util:
            if self.cpu_util > target.cpu_util:
                return False

        if self.gpu_util and target.gpu_util:
            if self.gpu_util > target.gpu_util:
                return False

        if self.gpu_used_memory and target.gpu_used_memory:
            if self.gpu_used_memory > target.gpu_used_memory:
                return False

        return True


class PerformanceObjectives(BaseModel):
    # Objectives values indicate the relative importance of each of criteria
    perf_throughput: Optional[float] = 0
    perf_latency_p99: Optional[int] = 0
    gpu_used_memory: Optional[int] = 0
    gpu_free_memory: Optional[int] = 0
    gpu_util: Optional[int] = 0
    cpu_used_memory: Optional[int] = 0
    cpu_free_memory: Optional[int] = 0
    cpu_util: Optional[int] = 0


class PerformanceTargets(BaseModel):
    constraints: Optional[Metrics] = None
    objectives: Optional[PerformanceObjectives] = None


class ModelLoadRequest(BaseModel):
    model_name: str
    load_type: LoadType
    method: ConfigGeneration
    batch_size: int = 1
    perf_targets: PerformanceTargets
