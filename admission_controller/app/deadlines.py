import logging
from collections import defaultdict
from fractions import Fraction
from operator import attrgetter
from typing import Dict, List, Tuple

import numpy as np
from recordclass import RecordClass

logger = logging.getLogger(__name__)
logger.propagate = False


class Deadlines:
    class ModelInfo(RecordClass):
        name: str
        period: float
        latency: float
        max_model_cpu_util: float

    @staticmethod
    def work_interval_gen(model_infos: List[ModelInfo], hyperperiod):
        index = 0
        for model in model_infos:
            for i in np.arange(start=0, stop=hyperperiod, step=model.period):
                yield (
                    model.name,
                    i,
                    i + model.latency,
                    i + model.period,
                    model.max_model_cpu_util,
                    model.latency * model.max_model_cpu_util,
                    index,
                )
                index += 1

    @staticmethod
    def get_hyperperiod(periods: List[float]) -> float:
        """
        Generate the hyperperiod (lcm) of the set of model requests to be scheduled

        Returns:
            int: hyperperiod
        """
        fraction_periods = np.array([Fraction(x) for x in periods])
        # If hyperperiod ends up very large,
        # just take the max (10x the largest period, the computed hyperperiod, and 10)
        return float(
            min(
                10 * max(fraction_periods),
                np.lcm.reduce(fraction_periods),
            )
        )

    @staticmethod
    def predict_deadline_misses(
        model_infos: List[ModelInfo],
        available_cpus: int,
        hyperperiod: float,
    ) -> Tuple[Dict[str, int], float]:
        """
        Predicts the deadline miss rates within one hyperperiod for all active models
        in the system assuming worst case scheduling scenario.
        We assume in this algorithm that scheduling is fair among the models
        running concurrently in the system

        Args:
            model_infos (List[ModelInfo]): List of active model information to be used in deadline projections
            available_cpus (int): number of cpus available to Triton server
            hyperperiod (float): Hyperperiod of model requests

        Returns:
            Tuple[Dict[str, int], float]: Dict containing num deadline misses per model and the proj cpu util
        """

        # Initialize beginning of simulated schedule based on current requests
        current_time = 0.0

        work_intervals = np.fromiter(
            Deadlines.work_interval_gen(model_infos, hyperperiod),
            dtype=[
                ("model_name", "<U100"),
                ("start_time", "<f4"),
                ("end_time", "<f4"),
                ("deadline", "<f4"),
                ("max_cpu_util", "<f4"),
                ("cpu_seconds", "<f4"),
                ("index", "<i4"),
            ],
        )

        logger.info(f"Hyperperiod: {hyperperiod}")
        logger.info(f"Number of work intervals: {len(work_intervals)}")
        logger.info(f"Avail cpus: {available_cpus}")

        # Iterate through the schedule for all active models until the next event
        # (inference stop or start) in the system is beyond the hyperperiod
        while current_time < hyperperiod:

            # Determine whether inference requests are overlapping in current time interval
            # which in this case is when there are multiple inferences running concurrently
            active_inference_intervals_index = np.where(
                (current_time >= work_intervals["start_time"])
                & (current_time < work_intervals["end_time"])
            )
            active_inference_intervals = work_intervals[
                active_inference_intervals_index
            ]
            upcoming_intervals = work_intervals[
                np.where(work_intervals["start_time"] > current_time)
            ]
            if (
                len(active_inference_intervals) == 0
                and len(upcoming_intervals) == 0
            ):
                break

            # Compute the work interval which will finish next if any
            try:
                next_completed_interval = active_inference_intervals[
                    np.argmin(active_inference_intervals["end_time"])
                ]
            except ValueError:
                next_completed_interval = None

            # Compute a global slowdown factor using the model profiled
            # cpu utilization
            max_cpu_util_projected = np.sum(
                active_inference_intervals["max_cpu_util"]
            )

            # Slowdown factor computed as a combination of max cpu utilization going over
            # total available in the system and the current slowdown measured by the system in
            # realtime
            slowdown_factor = max(
                (max_cpu_util_projected / (available_cpus * 100)), 1
            )

            # Compute the next key event. Either an inference finishes, or a new inference starts
            if len(upcoming_intervals) > 0:
                next_start_interval = upcoming_intervals[
                    np.argmin(upcoming_intervals["start_time"] - current_time)
                ]
                if next_completed_interval:
                    next_key_event_time = min(
                        next_completed_interval["end_time"] * slowdown_factor,
                        next_start_interval["start_time"],
                    )
                else:
                    next_key_event_time = next_start_interval["start_time"]
            elif next_completed_interval:
                next_key_event_time = (
                    next_completed_interval["end_time"] * slowdown_factor
                )

            # We approximate slowdown in this case by assuming all concurrent inferences share system resources
            # fairly within the linux scheduler
            # Update finish time of interval accounting for slowdown

            if slowdown_factor > 1:
                logger.debug(slowdown_factor)
                compute_duration = next_key_event_time - current_time
                np.add.at(
                    work_intervals["end_time"],
                    active_inference_intervals_index,
                    compute_duration - (compute_duration / slowdown_factor),
                )

            current_time = next_key_event_time

        # Once finalized work intervals computed, now compute deadline misses by model
        # Also Compute total_cpu_util using work intervals
        miss_counts: Dict[str, int] = defaultdict(int)

        cpu_seconds = np.sum(work_intervals["cpu_seconds"])

        missed_intervals = work_intervals[
            np.where(work_intervals["end_time"] > work_intervals["deadline"])
        ]
        for missed_interval in missed_intervals:
            miss_counts[missed_interval["model_name"]] += 1

        return dict(miss_counts), (
            cpu_seconds / (hyperperiod * available_cpus)
        )
