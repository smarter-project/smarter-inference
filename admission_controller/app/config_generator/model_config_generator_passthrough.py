import logging
import multiprocessing
from typing import Dict, Tuple

import GPUtil
from admission_controller.app.constants import SCALING_FACTOR
from model_analyzer.triton.model.model_config import ModelConfig

from ..admission_controller_exceptions import AdmissionControllerException
from ..model import *
from .model_config_generator import ModelConfigGenerator

logger = logging.getLogger(__name__)
logger.propagate = False


class ModelConfigGeneratorPassthrough(ModelConfigGenerator):
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
        if not (
            self._load_request.load_type
            in [LoadType.existing, LoadType.auto_gen]
        ):
            raise AdmissionControllerException(
                "Passthrough only supports 'existing' and 'auto_gen' load type"
            )

        if self._load_request.load_type == LoadType.auto_gen:
            has_gpu = bool(GPUtil.getAvailable())
            model_config_dict[
                "instance_group"
            ] = ModelConfigGenerator.instance_group_string_to_config(
                "1/GPU" if has_gpu else "1/CPU"
            )

            if self._model_type == ModelType.tf:
                model_config_dict["parameters"] = {
                    "TF_NUM_INTRA_THREADS": {
                        "string_value": str(
                            int(multiprocessing.cpu_count() / 2)
                        )
                    },
                    "TF_NUM_INTER_THREADS": {
                        "string_value": str(
                            int(multiprocessing.cpu_count() / 2)
                        )
                    },
                }
                try:
                    if has_gpu:
                        model_config_dict["optimization"] = {
                            "execution_accelerators": {
                                "gpu_execution_accelerator": [
                                    {
                                        "name": "tensorrt",
                                        "parameters": {
                                            "precision_mode": "FP16"
                                        },
                                    }
                                ]
                            }
                        }
                except:
                    logger.warning(
                        "Tensorrt was not profiled in this perf data"
                    )

            elif self._model_type == ModelType.tflite:
                model_config_dict["parameters"]["tflite_num_threads"] = {
                    "string_value": str(int(multiprocessing.cpu_count() / 2))
                }

                model_config_dict["optimization"] = {
                    "execution_accelerators": {
                        "cpu_execution_accelerator": [
                            {
                                "name": "xnnpack",
                                "parameters": {
                                    "num_threads": str(
                                        int(multiprocessing.cpu_count() / 2)
                                    ),
                                },
                            }
                        ]
                    }
                }

        # We have no idea what the perforance is going to be so just return
        # some dummy values
        logger.info(f"Predicted model mem: n/a")

        metrics_projected = Metrics(
            perf_throughput=1 / SCALING_FACTOR,
            perf_latency_p99=100 * SCALING_FACTOR,
            cpu_used_memory=0.0,
            cpu_util=0.0,
            max_cpu_util=0.0,
            system_cpu_util=0.0,
        )

        return model_config_dict, metrics_projected, 0
