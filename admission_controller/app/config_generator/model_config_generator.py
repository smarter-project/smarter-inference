import logging
import os
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from fractions import Fraction
from math import exp
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error

from ..admission_controller_exceptions import AdmissionControllerException
from ..constants import *
from ..deadlines import Deadlines
from ..model import *
from ..model_state import ModelState

logger = logging.getLogger(__name__)
logger.propagate = False


class ModelConfigGenerator(ABC):
    """
    Defines the interface for handling load requests from the
    admission controller
    """

    def __init__(
        self,
        load_request: ModelLoadRequest,
        system_metrics_actual: Metrics,
        current_model_states: Dict[str, ModelState],
        hyperperiod: float,
    ) -> None:
        self._load_request = load_request
        self._model_dir = os.path.join(
            OUTPUT_REPO_PATH, load_request.model_name
        )
        self._model_perf_file_path = os.path.join(self._model_dir, "perf.csv")
        self._system_metrics_actual = system_metrics_actual
        self._model_name = load_request.model_name
        self._current_model_states = current_model_states
        self._model_type = self.get_model_type(self._model_dir)
        self._hyperperiod = hyperperiod
        self._linear_mem_models: Dict[str, LinearRegression] = {}

        # Generate the perf df for the model from the profiled data
        (_, _, self._perf_df,) = self.model_analyzer_csv_to_df(
            self._model_type,
            self._model_perf_file_path,
            perf_constraints=self._current_model_states[
                self._model_name
            ].perf_constraints,
        )
        self._gen_mem_linreg_models()

    @abstractmethod
    def process_triton_model_configuration(
        self,
    ) -> Tuple[Dict, Metrics, int]:
        """
        Generates or verifies existing triton model configuration and predicts resource util
        """
        pass

    @staticmethod
    def get_model_type(model_dir: str):
        """
        Use the model directory to get the model type based on file extension
        """
        for (_, _, filenames) in os.walk(model_dir + "/1"):
            for filename in filenames:
                file_extension = pathlib.Path(filename).suffix
                if file_extension == ".tflite":
                    return ModelType.tflite
                elif file_extension == ".graphdef":
                    return ModelType.tf

        raise AdmissionControllerException("Model file not found in directory")

    def _compute_projected_single_model_cpu_utilization(
        self, profiled_latency: float, profiled_cpu_util: float = None
    ) -> float:
        """
        Computes the projected CPU utilization for a given profiled latency and cpu utilization.
        Latency in miliseconds
        """

        period = (
            1
            / self._current_model_states[
                self._model_name
            ].min_throughput_constraint
        )
        latency = profiled_latency / 1000

        # This calculation assumes the computation takes 100% of every core in the system
        projected_cpu_utilization = latency / period

        # Scale cpu utilization for better estimate if profiled data available, which accounts for inference
        # cpu utilization when latency = period
        if profiled_cpu_util:
            scale_cpu_coef = profiled_cpu_util / (MAX_CPU_PERCENT)
            projected_cpu_utilization *= scale_cpu_coef

        return projected_cpu_utilization

    def _gen_mem_linreg_models(self):
        self._x_names = [
            "Batch",
            "Concurrency",
            "CPU",
            "GPU",
        ]
        if self._model_type == ModelType.tflite:
            self._x_names.extend(
                [
                    "backend_parameter/tflite_num_threads",
                    "cpu_accelerator_num_threads",
                ]
            )

        elif self._model_type == ModelType.tf:
            self._x_names.extend(
                [
                    "backend_parameter/TF_NUM_INTRA_THREADS",
                    "backend_parameter/TF_NUM_INTER_THREADS",
                    "tensorrt",
                ]
            )

        # Generate linear regression model for mem
        if self._model_type == ModelType.tflite:
            for cpu_accelerator in self._perf_df["cpu_accelerator"].unique():
                df_by_accelerator = self._perf_df[
                    self._perf_df["cpu_accelerator"] == cpu_accelerator
                ]
                X = df_by_accelerator.loc[:, self._x_names]
                y = df_by_accelerator.loc[:, "RAM Usage (MB)"]
                self._linear_mem_models[
                    cpu_accelerator
                ] = LinearRegression().fit(X, y)
                score = self._linear_mem_models[cpu_accelerator].score(X, y)
                self._max_err = max_error(
                    y, self._linear_mem_models[cpu_accelerator].predict(X)
                )
                logger.info(f"Max Error {cpu_accelerator}: {self._max_err}")
                logger.info(
                    f"Regression score {cpu_accelerator}: {score}, intercept:"
                    f" {self._linear_mem_models[cpu_accelerator].intercept_},"
                    f" coef: {self._linear_mem_models[cpu_accelerator].coef_}"
                )
                if score < 0.65 and not (cpu_accelerator == "default"):
                    raise AdmissionControllerException(
                        "Regression score for cpu_accelerator"
                        f" {cpu_accelerator} to memory too low: {score}"
                    )
        elif self._model_type == ModelType.tf:
            X = self._perf_df.loc[:, self._x_names]
            y = self._perf_df.loc[:, "RAM Usage (MB)"]
            self._linear_mem_models["default"] = LinearRegression().fit(X, y)
            score = self._linear_mem_models["default"].score(X, y)
            self._max_err = max_error(
                y, self._linear_mem_models["default"].predict(X)
            )
            logger.info(f"Max Error: {self._max_err}")
            logger.info(
                f"Regression score: {score}, intercept:"
                f" {self._linear_mem_models['default'].intercept_},"
                f" coef: {self._linear_mem_models['default'].coef_}"
            )
            if score < 0.65:
                raise AdmissionControllerException(
                    "Regression score for cpu_accelerator"
                    f" default to memory too low: {score}"
                )

    def _compute_projected_single_model_res_mem_utilization_tflite(
        self,
        cpu_instances: int,
        gpu_instances: int = 0,
        accelerator_name: str = "default",
        concurrency: int = 1,
        batch_size: int = 1,
        tflite_default_threads: int = 1,
        accelerator_threads: int = 1,
    ) -> Tuple[float, Optional[float]]:
        """
        Computes the projected resident mem size for a given model
        """
        if type(accelerator_threads) == str:
            accelerator_threads = 0
        else:
            accelerator_threads = int(accelerator_threads)

        # First attempt to find profiled mem consumption, if not found fall back
        # to the linear regression model to predict the memory
        try:
            df = self._perf_df.loc[
                (self._perf_df["Batch"] == batch_size)
                & (self._perf_df["Concurrency"] == concurrency)
                & (self._perf_df["CPU"] == cpu_instances)
                & (self._perf_df["GPU"] == gpu_instances)
                & (self._perf_df["cpu_accelerator"] == accelerator_name)
                & (
                    self._perf_df["backend_parameter/tflite_num_threads"]
                    == tflite_default_threads
                )
                & (
                    self._perf_df["cpu_accelerator_num_threads"]
                    == accelerator_threads
                )
            ]
            return df.iloc[0]["RAM Usage (MB)"], None
        except IndexError:
            logger.info(
                "Could not lookup mem consumption, falling back to lin model"
            )
            input_data = pd.DataFrame(
                {
                    "Batch": [batch_size],
                    "Concurrency": [concurrency],
                    "CPU": [cpu_instances],
                    "GPU": [gpu_instances],
                    "backend_parameter/tflite_num_threads": [
                        tflite_default_threads
                    ],
                    "cpu_accelerator_num_threads": [accelerator_threads],
                }
            )
            input_data = input_data.apply(pd.to_numeric, errors="coerce")
            input_data.fillna(0, inplace=True)

            prediction = self._linear_mem_models[accelerator_name].predict(
                input_data
            )[0]

            return (
                prediction,
                self._linear_mem_models[accelerator_name].intercept_,
            )

    def _compute_projected_single_model_res_mem_utilization_tf(
        self,
        cpu_instances: int,
        gpu_instances: int = 0,
        tensorrt: int = 0,
        concurrency: int = 1,
        batch_size: int = 1,
        tf_intra_threads: int = 1,
        tf_inter_threads: int = 1,
    ) -> Tuple[float, Optional[float]]:
        """
        Computes the projected resident mem size for a given model
        """

        # First attempt to find profiled mem consumption, if not found fall back
        # to the linear regression model to predict the memory
        try:
            df = self._perf_df.loc[
                (self._perf_df["Batch"] == batch_size)
                & (self._perf_df["Concurrency"] == concurrency)
                & (self._perf_df["CPU"] == cpu_instances)
                & (self._perf_df["GPU"] == gpu_instances)
                & (self._perf_df["tensorrt"] == tensorrt)
                & (
                    self._perf_df["backend_parameter/TF_NUM_INTRA_THREADS"]
                    == tf_intra_threads
                )
                & (
                    self._perf_df["backend_parameter/TF_NUM_INTER_THREADS"]
                    == tf_inter_threads
                )
            ]
            logger.info(df["RAM Usage (MB)"])
            return df.iloc[0]["RAM Usage (MB)"], None
        except IndexError as e:
            logger.info(
                "Could not lookup mem consumption, falling back to lin model"
            )

            input_data = pd.DataFrame(
                {
                    "Batch": [batch_size],
                    "Concurrency": [concurrency],
                    "CPU": [cpu_instances],
                    "GPU": [gpu_instances],
                    "backend_parameter/TF_NUM_INTRA_THREADS": [
                        tf_intra_threads
                    ],
                    "backend_parameter/TF_NUM_INTER_THREADS": [
                        tf_inter_threads
                    ],
                    "tensorrt": [tensorrt],
                }
            )

            input_data = input_data.apply(pd.to_numeric, errors="coerce")
            input_data.fillna(0, inplace=True)

            prediction = self._linear_mem_models["default"].predict(
                input_data
            )[0]

            return (
                prediction,
                self._linear_mem_models["default"].intercept_,
            )

    def _apply_runtime_related_config(self, model_config_dict: Dict):
        if self._model_type == ModelType.tf:
            try:
                model_config_dict["parameters"][
                    "TF_USE_PER_SESSION_THREADS"
                ] = {"string_value": "True"}
            except KeyError:
                model_config_dict["parameters"] = {
                    "TF_USE_PER_SESSION_THREADS": {"string_value": "True"}
                }

    @staticmethod
    def get_model_instances(instance_group_str) -> Tuple[int, int]:
        instance_groups = instance_group_str.split(",")

        cpu_instances = 0
        gpu_instances = 0

        for instance_group in instance_groups:
            tokens = instance_group.split("/")
            count = tokens[0]
            kind = tokens[1]
            if kind == "CPU":
                cpu_instances = int(count)
            elif kind == "GPU":
                gpu_instances = int(count)

        return cpu_instances, gpu_instances

    @staticmethod
    def update_model_config_from_df_row(
        model_config_dict: Dict,
        model_type: ModelType,
        row: Dict,
    ):
        # Overwrite model config keys with values from selected csv row
        for (
            key,
            value,
        ) in ModelConfigGenerator.get_model_config_params_from_df_row(
            model_type, row
        ).items():
            model_config_dict[key] = value

    @staticmethod
    def get_model_config_params_from_df_row(
        model_type: ModelType, row: Dict
    ) -> Dict:
        """
        Returns triton model config values from model analyzer csv row
        """
        model_config_params: defaultdict = defaultdict(dict)

        model_config_params[
            "instance_group"
        ] = ModelConfigGenerator.instance_group_string_to_config(
            f"{int(row['CPU'])}/{'CPU'},{int(row['GPU'])}/{'GPU'}"
        )

        if model_type == ModelType.tf:
            model_config_params["parameters"] = {
                "TF_NUM_INTRA_THREADS": {
                    "string_value": str(
                        int(row["backend_parameter/TF_NUM_INTRA_THREADS"])
                    )
                },
                "TF_NUM_INTER_THREADS": {
                    "string_value": str(
                        int(row["backend_parameter/TF_NUM_INTER_THREADS"])
                    )
                },
            }
            try:
                if row["tensorrt"] == 1:
                    model_config_params["optimization"] = {
                        "execution_accelerators": {
                            "gpu_execution_accelerator": [
                                {
                                    "name": "tensorrt",
                                    "parameters": {"precision_mode": "FP16"},
                                }
                            ]
                        }
                    }
            except:
                logger.warning("Tensorrt was not profiled in this perf data")

        elif model_type == ModelType.tflite:
            model_config_params["parameters"]["tflite_num_threads"] = {
                "string_value": str(
                    int(row["backend_parameter/tflite_num_threads"])
                )
            }

            if row["cpu_accelerator"] == "armnn":
                model_config_params["optimization"] = {
                    "execution_accelerators": {
                        "cpu_execution_accelerator": [
                            {
                                "name": "armnn",
                                "parameters": {
                                    "num_threads": row[
                                        "cpu_accelerator_num_threads"
                                    ],
                                    "reduce_fp32_to_fp16": row[
                                        "armnn_fp32_to_fp16"
                                    ],
                                    "reduce_fp32_to_bf16": row[
                                        "armnn_fp32_to_bf16"
                                    ],
                                    "fast_math_enabled": row[
                                        "armnn_fast_math_enabled"
                                    ],
                                },
                            }
                        ]
                    }
                }

            elif row["cpu_accelerator"] == "xnnpack":
                model_config_params["optimization"] = {
                    "execution_accelerators": {
                        "cpu_execution_accelerator": [
                            {
                                "name": "xnnpack",
                                "parameters": {
                                    "num_threads": str(
                                        int(row["cpu_accelerator_num_threads"])
                                    ),
                                },
                            }
                        ]
                    }
                }

            elif row["cpu_accelerator"] == "default":
                model_config_params["optimization"] = {}

        # Get preferred batch size from csv file string
        # model_config_params["dynamic_batching"]["preferred_batch_size"] = ModelConfigGenerator.preferred_batch_size_string_to_config(
        #     row["Preferred Batch Sizes"])

        return dict(model_config_params)

    @staticmethod
    def cost(perf_constraints: Metrics, achieved_perf: Metrics):
        """
        Compute cost of achieved perf from perf model data
        Cost is measured as delta from objectives

        Cost is meausred as: (num_constraints) * (logical nand of all constraints met) * (1 - (2 / (1 + e^3x)))
        where x is %diff of actual vs constraints. This function is asymptotically bounded at
        1 when x > 0, so 1 can be max penalty given per constraint.

        Note that VW expects only to minimize cost function, so smaller numbers mean larger reward
        """

        costs = {}

        # Calculate whether constraints were met by current measurement
        constraints_met = True
        if perf_constraints:
            # Compare achieved perf against the perf constraints specified in model state
            constraints_met = achieved_perf.constraints_met(perf_constraints)

        if constraints_met:
            # Measure the percent difference between the current measurements
            # and the performance constraints weighed by objective importance
            for k, v in perf_constraints.dict().items():
                if v:
                    achieved = achieved_perf.dict()[k]
                    target = float(v)

                    percent_diff = abs(achieved - target) / (
                        (achieved + target) / 2.0
                    )

                    costs[k] = 1 - (2 / (1 + exp(3 * percent_diff)))

            cost = sum(costs.values())
        else:
            # Max cost computed as length of set constraints
            cost = len([x for x in perf_constraints.dict().values() if x])

        return cost

    @staticmethod
    def instance_group_string_to_config(instance_group_string: str) -> List:
        """
        Parameters
        ----------
        instance_group_string : str
            dynamic batching string

        Returns
        -------
        list
            instance group list from input string
        """

        result = []

        instance_groups = instance_group_string.split(",")

        for instance_group in instance_groups:
            tokens = instance_group.split("/")
            count = tokens[0]
            kind = tokens[1]
            if int(count) > 0:
                result.append({"count": count, "kind": f"KIND_{kind}"})

        return result

    @staticmethod
    def preferred_batch_size_string_to_config(
        preferred_batch_size_string: str,
    ) -> Optional[List]:
        """
        Parameters
        ----------
        preferred_batch_size_string : str
            preferred batch size string

        Returns
        -------
        list
            dynamic batching dict from input string
        """
        if preferred_batch_size_string in ["Disabled", "N/A"]:
            return None

        return preferred_batch_size_string.strip("][").split(" ")

    def _deadline_misses_from_row(self, row: pd.DataFrame):
        model_infos = []
        # Prepare model information to submit to deadline miss projection algorithm
        for model_name, model_state in self._current_model_states.items():
            period = Fraction(1 / model_state.min_throughput_constraint)
            if model_name == self._model_name:
                latency = (float(row["p99 Latency (ms)"])) / 1000
                max_model_cpu_util = row["CPU Utilization (%)"]
            else:
                # Use realtime latency in projection if available
                try:
                    slowdown = model_state.triton_stats["slowdown"]
                except KeyError as e:
                    slowdown = 1.0

                if slowdown > 1.0:
                    logger.info(f"Using slowdown of {slowdown}")
                    latency = (
                        (model_state.metrics_projected.perf_latency_p99)
                        * slowdown
                        / 1000
                    )
                else:
                    latency = (
                        model_state.metrics_projected.perf_latency_p99
                    ) / 1000

                max_model_cpu_util = model_state.metrics_projected.max_cpu_util
            model_infos.append(
                Deadlines.ModelInfo(
                    name=model_name,
                    latency=latency,
                    period=period,
                    max_model_cpu_util=max_model_cpu_util,
                )
            )
        (deadline_misses, cpu_util,) = Deadlines.predict_deadline_misses(
            model_infos, NUM_TRITON_CPUS, self._hyperperiod
        )

        return pd.Series(
            [
                sum(dict(deadline_misses).values()),
                cpu_util,
            ]
        )

    def _add_deadline_misses_to_perf_df(self, df: pd.DataFrame):
        df[["deadline_misses", "total_system_cpu_util_projected"]] = df.apply(
            self._deadline_misses_from_row,
            axis=1,
        )

    @staticmethod
    def model_analyzer_csv_to_df(
        model_type: ModelType,
        csv_filepath: str,
        perf_constraints: Optional[Metrics] = None,
    ) -> Tuple[List[str], List[str], pd.DataFrame]:
        """
        Convert perf csv file to df

        Args:
            csv_filepath : str
                Path to model analyzer generated csv file

        Returns:
            Tuple[List, List, Dataframe]: tuple containing x_names, y_names and
                resultant dataframe
        """
        x_names = [
            "Batch",
            "Concurrency",
            "Model Config Path",
            "Instance Group",
        ]
        y_names = [
            "Throughput (infer/sec)",
            "p99 Latency (ms)",
            "RAM Usage (MB)",
            "CPU Utilization (%)",
        ]

        if model_type == ModelType.tf:
            x_names.extend(
                [
                    "backend_parameter/TF_NUM_INTRA_THREADS",
                    "backend_parameter/TF_NUM_INTER_THREADS",
                    "tensorrt",
                ]
            )
            y_names.extend(["GPU Utilization (%)"])
        if model_type == ModelType.tflite:
            x_names.extend(
                [
                    "backend_parameter/tflite_num_threads",
                    "cpu_accelerator",
                    "cpu_accelerator_num_threads",
                    "armnn_fp32_to_fp16",
                    "armnn_fp32_to_bf16",
                    "armnn_fast_math_enabled",
                ]
            )
        names = x_names + y_names

        no_gpu_data = False
        dataframe: pd.DataFrame = pd.DataFrame()
        try:
            dataframe = pd.read_csv(csv_filepath, usecols=names)
        except ValueError:
            if model_type == ModelType.tf:
                names.remove("tensorrt")
                names.remove("GPU Utilization (%)")
                dataframe = pd.read_csv(csv_filepath, usecols=names)
                no_gpu_data = True

        # Create new columns in CSV file to account for CPU and GPU instances in int form
        dataframe["CPU"] = dataframe.apply(
            lambda row: int(row["Instance Group"].split("/")[0])
            if row["Instance Group"].split("/")[1] == "CPU"
            else 0,
            axis=1,
        )
        dataframe["GPU"] = dataframe.apply(
            lambda row: int(row["Instance Group"].split("/")[0])
            if row["Instance Group"].split("/")[1] == "GPU"
            else 0,
            axis=1,
        )
        if no_gpu_data:
            dataframe["tensorrt"] = 0
            dataframe["GPU Utilization (%)"] = 0.0
        dataframe.drop(labels="Instance Group", axis=1, inplace=True)
        x_names.remove("Model Config Path")
        x_names.remove("Instance Group")
        x_names.extend(["CPU", "GPU"])

        # Remove all rows from dataframe with default config
        dataframe = dataframe[
            dataframe["Model Config Path"].str.contains("default") == False
        ]

        dataframe.drop(labels="Model Config Path", axis=1, inplace=True)

        if model_type == ModelType.tf:
            dataframe = dataframe.astype(
                {"backend_parameter/TF_NUM_INTRA_THREADS": "int32"}
            )
            dataframe = dataframe.astype(
                {"backend_parameter/TF_NUM_INTER_THREADS": "int32"}
            )
            if perf_constraints:
                dataframe["cost"] = dataframe.apply(
                    lambda row: ModelConfigGenerator.cost(
                        perf_constraints,
                        Metrics(
                            perf_throughput=float(
                                row["Throughput (infer/sec)"]
                            ),
                            perf_latency_p99=float(row["p99 Latency (ms)"]),
                            cpu_used_memory=float(row["RAM Usage (MB)"]),
                            cpu_util=float(row["CPU Utilization (%)"]),
                            gpu_util=float(row["GPU Utilization (%)"]),
                        ),
                    ),
                    axis=1,
                )

        elif model_type == ModelType.tflite:
            dataframe["cpu_accelerator_num_threads"] = dataframe[
                "cpu_accelerator_num_threads"
            ].apply(pd.to_numeric, errors="coerce")
            dataframe["cpu_accelerator_num_threads"].fillna(0, inplace=True)
            dataframe = dataframe.astype(
                {"cpu_accelerator_num_threads": "int32"}
            )
            dataframe = dataframe.astype(
                {"backend_parameter/tflite_num_threads": "int32"}
            )

            if perf_constraints:
                dataframe["cost"] = dataframe.apply(
                    lambda row: ModelConfigGenerator.cost(
                        perf_constraints,
                        Metrics(
                            perf_throughput=float(
                                row["Throughput (infer/sec)"]
                            ),
                            perf_latency_p99=float(row["p99 Latency (ms)"]),
                            cpu_used_memory=float(row["RAM Usage (MB)"]),
                            cpu_util=float(row["CPU Utilization (%)"]),
                        ),
                    ),
                    axis=1,
                )

        return x_names, y_names, dataframe
