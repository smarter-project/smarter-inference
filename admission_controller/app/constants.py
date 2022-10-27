import multiprocessing
import os
from typing import Optional

from cpuset.cset import CpuSet


def cpu_list_from_cpu_str(cpu_str: Optional[str]):
    if not (cpu_str):
        avail_cpus = list(range(0, multiprocessing.cpu_count()))
    else:
        avail_cpus = []
        cpu_list = cpu_str.split(",")
        for cpu_range in cpu_list:
            if "-" in cpu_range:
                range_pair = cpu_range.split("-")
                avail_cpus.extend(
                    list(range(int(range_pair[0]), int(range_pair[1]) + 1))
                )
            else:
                avail_cpus.extend([int(cpu_range)])
    return avail_cpus


# Triton Constants
SHIELD = bool(os.getenv("SHIELD"))
if SHIELD:
    # Get triton cpus from cpuset information
    cpuset = CpuSet()
    cpuset.read_cpuset("/user")
    AVAILABLE_TRITON_CPUS = cpu_list_from_cpu_str(cpuset.getcpus())
else:
    AVAILABLE_TRITON_CPUS = cpu_list_from_cpu_str(os.getenv("TRITON_CPUS"))

try:
    GPUS = [int(x) for x in os.getenv("GPUS", "").split(",")]
except ValueError:
    GPUS = []

SCHED = os.getenv("SCHED")

MULTI_SERVER = bool(os.getenv("MULTI_SERVER"))
NUM_TRITON_CPUS = len(AVAILABLE_TRITON_CPUS)
TRITON_INSTALL_PATH = os.getenv("TRITON_INSTALL_PATH", "/opt/tritonserver")
OUTPUT_REPO_PATH = os.getenv("OUTPUT_MODEL_REPO_PATH", "/opt/output_models")
CLIENT_MAX_RETRIES = int(os.getenv("CLIENT_MAX_RETRIES", 20))
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
PRIORITY_ASSIGNMENT = os.getenv("PRIORITY_ASSIGNMENT")


# API Server Constants
SCALING_FACTOR = float(os.getenv("SCALING_FACTOR", "1.0"))
NO_ENFORCE = bool(os.getenv("NO_ENFORCE"))
MAX_CPU_PERCENT = NUM_TRITON_CPUS * 100

# Nginx Constants
NGINX_LOGGING = bool(os.getenv("NGINX_LOGGING"))
NGINX_CONFIG_PATH = os.getenv("NGINX_CONFIG_PATH", "/tmp/nginx_config.txt")
NO_NGINX_RATELIMIT = bool(os.getenv("NO_NGINX_RATELIMIT"))
