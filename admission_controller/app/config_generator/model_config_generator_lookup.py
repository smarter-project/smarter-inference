import logging
from typing import Dict, Tuple

from admission_controller.app.constants import (
    MISS_TOLERANCE,
    NO_ENFORCE_ADMISSION_CONTROL,
    NUM_TRITON_CPUS,
    SCALING_FACTOR,
)
from model_analyzer.triton.model.model_config import ModelConfig

from ..admission_controller_exceptions import (
    AdmissionControllerException,
    NoPerfDataFoundException,
)
from ..model import *
from .model_config_generator import ModelConfigGenerator

logger = logging.getLogger(__name__)
logger.propagate = False


class ModelConfigGeneratorLookup(ModelConfigGenerator):
    def process_triton_model_configuration(
        self,
    ) -> Tuple[Dict, Metrics, int]:
        """
        Use only the profiled perf data to select a configuration. Closest data point will be used
        to generate the model configuration

        Returns model config dict
        """

        # Generate model config from dict if there was an optimal row, else raise
        model_config_dict = ModelConfig.create_from_file(
            self._model_dir
        ).get_config()

        # Lookup matching row based on perf data
        if self._load_request.load_type == LoadType.existing:
            row = self._validate_existing_config(model_config_dict)
        else:
            row = self._generate_config()

        self.update_model_config_from_df_row(
            model_config_dict, self._model_type, row
        )
        self._apply_runtime_related_config(model_config_dict)

        if self._model_type == ModelType.tflite:
            (
                cpu_used_memory,
                triton_mem,
            ) = self._compute_projected_single_model_res_mem_utilization_tflite(
                cpu_instances=int(row["CPU"]),
                accelerator_name=row["cpu_accelerator"],
                batch_size=int(row["Batch"]),
                concurrency=int(row["Concurrency"]),
                tflite_default_threads=int(
                    row["backend_parameter/tflite_num_threads"]
                ),
                accelerator_threads=int(row["cpu_accelerator_num_threads"]),
            )
        elif self._model_type == ModelType.tf:
            (
                cpu_used_memory,
                triton_mem,
            ) = self._compute_projected_single_model_res_mem_utilization_tf(
                cpu_instances=int(row["CPU"]),
                gpu_instances=int(row["GPU"]),
                batch_size=int(row["Batch"]),
                concurrency=int(row["Concurrency"]),
                tensorrt=int(row["tensorrt"]),
                tf_inter_threads=int(
                    row["backend_parameter/TF_NUM_INTER_THREADS"]
                ),
                tf_intra_threads=int(
                    row["backend_parameter/TF_NUM_INTRA_THREADS"]
                ),
            )
        logger.info(f"Predicted model mem: {cpu_used_memory}")

        metrics_projected = Metrics(
            perf_throughput=float(row["Throughput (infer/sec)"])
            / SCALING_FACTOR,
            perf_latency_p99=float(row["p99 Latency (ms)"]) * SCALING_FACTOR,
            cpu_used_memory=cpu_used_memory,
            cpu_util=self._compute_projected_single_model_cpu_utilization(
                float(row["p99 Latency (ms)"]) * SCALING_FACTOR,
                profiled_cpu_util=float(row["CPU Utilization (%)"]),
            ),
            max_cpu_util=float(row["CPU Utilization (%)"]),
            system_cpu_util=float(row["total_system_cpu_util_projected"]),
        )

        return model_config_dict, metrics_projected, row["deadline_misses"]

    def _generate_config(self):
        df = self._perf_df.loc[
            (
                self._perf_df["Batch"]
                == self._current_model_states[
                    self._model_name
                ].max_requested_batch_size
            )
            & (
                self._perf_df["Concurrency"]
                == self._current_model_states[
                    self._model_name
                ].concurrent_clients
            )
            & (
                self._perf_df["RAM Usage (MB)"]
                < self._system_metrics_actual.cpu_free_memory
            )
            & (
                self._perf_df["CPU"] + self._perf_df["GPU"]
                <= self._current_model_states[
                    self._model_name
                ].concurrent_clients
            )
        ]

        if df.empty:
            raise AdmissionControllerException(
                "No valid perf data found for request"
            )
        logger.info(f"\n{df.to_string()}")

        self._add_deadline_misses_to_perf_df(df)

        if df.empty:
            raise AdmissionControllerException(
                "No configuration found where no missed deadlines"
            )

        df = df.sort_values(by=["deadline_misses", "cost"])
        logger.info(f"\n{df.to_string()}")

        if df.iloc[0]["cost"] == 1 * len(
            self._current_model_states[
                self._model_name
            ].perf_constraints.dict()
        ):
            raise AdmissionControllerException("No row fit constraints")

        if not (NO_ENFORCE_ADMISSION_CONTROL):
            df_no_misses = df.loc[(df["deadline_misses"] == 0)]
            if df_no_misses.empty:
                reason = (
                    "New load request failed to schedule within miss"
                    f" tolerance. Model: {self._model_name}, Misses:"
                    f" {df['deadline_misses'].to_string()} > tolerance:"
                    f" {self._miss_tolerance}"
                )
                raise AdmissionControllerException(reason)

        return df.iloc[0]

    def _validate_existing_config(self, model_config_dict: Dict):
        """
        Opens current model config sent by user and determines based on profiling data whether the model
        with serving config will fit in system

        Raises:
            AdmissionControllerException: if matching row not found

        Returns:
            Dict: matching row from profiled data
        """

        cpu_instances = 0
        gpu_instances = 0
        if "instance_group" in model_config_dict:
            if (
                str(model_config_dict["instance_group"][0]["kind"])
                == "KIND_CPU"
            ):
                cpu_instances = int(
                    model_config_dict["instance_group"][0]["count"]
                )
            else:
                gpu_instances = int(
                    model_config_dict["instance_group"][0]["count"]
                )
        else:
            cpu_instances = 1

        df = self._perf_df.loc[
            (
                self._perf_df["Batch"]
                == self._current_model_states[
                    self._model_name
                ].max_requested_batch_size
            )
            & (
                self._perf_df["Concurrency"]
                == self._current_model_states[
                    self._model_name
                ].concurrent_clients
            )
            & (self._perf_df["CPU"] == cpu_instances)
            & (self._perf_df["GPU"] == gpu_instances)
        ]
        if self._model_type == ModelType.tf:
            try:
                tensorrt = int(
                    model_config_dict["optimization"][
                        "execution_accelerators"
                    ]["gpu_execution_accelerator"][0]["name"]
                    == "tensorrt"
                    and gpu_instances > 0
                )

            except:
                tensorrt = 0

            try:
                tf_intra_threads = int(
                    model_config_dict["parameters"]["TF_NUM_INTRA_THREADS"][
                        "string_value"
                    ]
                )
                tf_inter_threads = int(
                    model_config_dict["parameters"]["TF_NUM_INTER_THREADS"][
                        "string_value"
                    ]
                )
            except KeyError:
                tf_intra_threads = NUM_TRITON_CPUS
                tf_inter_threads = tf_intra_threads

            df = df.loc[
                (df["tensorrt"] == tensorrt)
                & (
                    df["backend_parameter/TF_NUM_INTRA_THREADS"]
                    == tf_intra_threads
                )
                & (
                    df["backend_parameter/TF_NUM_INTER_THREADS"]
                    == tf_inter_threads
                )
            ]

        elif self._model_type == ModelType.tflite:
            try:
                tflite_default_threads = int(
                    model_config_dict["parameters"]["tflite_num_threads"][
                        "string_value"
                    ]
                )
            except KeyError:
                # This is the triton default for number of tflite threads if not provided
                tflite_default_threads = NUM_TRITON_CPUS

            # Cpu accelerator type
            try:
                accelerator_name = model_config_dict["optimization"][
                    "execution_accelerators"
                ]["cpu_execution_accelerator"][0]["name"]

                if accelerator_name == "xnnpack":
                    try:
                        accelerator_threads = int(
                            model_config_dict["optimization"][
                                "execution_accelerators"
                            ]["cpu_execution_accelerator"][0]["parameters"][
                                "num_threads"
                            ]
                        )
                    except KeyError:
                        accelerator_threads = NUM_TRITON_CPUS

            except KeyError:
                accelerator_name = "default"
                accelerator_threads = 0

            df = df.loc[
                (self._perf_df["cpu_accelerator"] == accelerator_name)
                & (
                    self._perf_df["backend_parameter/tflite_num_threads"]
                    == tflite_default_threads
                )
                & (
                    self._perf_df["cpu_accelerator_num_threads"]
                    == accelerator_threads
                )
            ]

        self._add_deadline_misses_to_perf_df(df)

        logger.info(df.to_string())

        if not (df.empty):
            if not (self._system_metrics_actual.cpu_free_memory):
                raise AdmissionControllerException("cpu free memory unknown")

            if self._model_type == ModelType.tflite:
                (
                    proj_mem,
                    triton_mem,
                ) = self._compute_projected_single_model_res_mem_utilization_tflite(
                    cpu_instances=cpu_instances,
                    accelerator_name=accelerator_name,
                    batch_size=self._current_model_states[
                        self._model_name
                    ].max_requested_batch_size,
                    concurrency=self._current_model_states[
                        self._model_name
                    ].concurrent_clients,
                    tflite_default_threads=tflite_default_threads,
                    accelerator_threads=accelerator_threads,
                )
            elif self._model_type == ModelType.tf:
                (
                    proj_mem,
                    triton_mem,
                ) = self._compute_projected_single_model_res_mem_utilization_tf(
                    cpu_instances=cpu_instances,
                    gpu_instances=gpu_instances,
                    batch_size=self._current_model_states[
                        self._model_name
                    ].max_requested_batch_size,
                    concurrency=self._current_model_states[
                        self._model_name
                    ].concurrent_clients,
                    tensorrt=tensorrt,
                    tf_inter_threads=df.iloc[0][
                        "backend_parameter/TF_NUM_INTER_THREADS"
                    ],
                    tf_intra_threads=df.iloc[0][
                        "backend_parameter/TF_NUM_INTRA_THREADS"
                    ],
                )
            proj_mem *= 1e6
            mem_fit = (proj_mem) < float(
                self._system_metrics_actual.cpu_free_memory
            )
            if not (mem_fit):
                raise AdmissionControllerException(
                    "Model did not fit in mem. Requested"
                    f" {proj_mem},"
                    f" had {self._system_metrics_actual.cpu_free_memory}"
                )
            if not (NO_ENFORCE_ADMISSION_CONTROL):
                df_no_misses = df.loc[(df["deadline_misses"] == 0)]
                if df_no_misses.empty:
                    reason = (
                        "New load request failed to schedule within miss"
                        f" tolerance. Model: {self._model_name}, Misses:"
                        f" {df['deadline_misses'].to_string()} > tolerance:"
                        f" {MISS_TOLERANCE}"
                    )
                    raise AdmissionControllerException(reason)
            return df.iloc[0]
        else:
            raise NoPerfDataFoundException("No perf data found for request")
