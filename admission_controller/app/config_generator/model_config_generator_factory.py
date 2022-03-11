from typing import Dict

from ..admission_controller_exceptions import AdmissionControllerException
from ..model import *
from ..model_state import ModelState
from .model_config_generator import ModelConfigGenerator
from .model_config_generator_lookup import ModelConfigGeneratorLookup


class ModelConfigGeneratorFactory:
    """
    A factory for creating model loader instances
    """

    @staticmethod
    def create_model_config_generator(
        load_request: ModelLoadRequest,
        system_metrics_actual: Metrics,
        current_model_states: Dict[str, ModelState],
        hyperperiod: float,
    ) -> ModelConfigGenerator:
        if load_request.method == ConfigGeneration.lookup:
            return ModelConfigGeneratorFactory.create_model_config_generator_lookup(
                load_request,
                system_metrics_actual,
                current_model_states,
                hyperperiod,
            )
        elif load_request.method == ConfigGeneration.lookup_lr:
            return ModelConfigGeneratorFactory.create_model_config_generator_lookup_lr(
                load_request,
                system_metrics_actual,
                current_model_states,
                hyperperiod,
            )
        else:
            raise AdmissionControllerException(
                "Config generation method unknown"
            )

    @staticmethod
    def create_model_config_generator_lookup(
        load_request: ModelLoadRequest,
        system_metrics_actual: Metrics,
        current_model_states: Dict[str, ModelState],
        hyperperiod: float,
    ):
        return ModelConfigGeneratorLookup(
            load_request,
            system_metrics_actual,
            current_model_states,
            hyperperiod,
        )

    @staticmethod
    def create_model_config_generator_lookup_lr(
        load_request: ModelLoadRequest,
        system_metrics_actual: Metrics,
        current_model_states: Dict[str, ModelState],
        hyperperiod: float,
    ):
        raise NotImplementedError
