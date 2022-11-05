# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import signal
import subprocess
from io import StringIO
from os import path
from time import time
from typing import Dict, List

import pandas as pd


class PerfStats:
    def __init__(
        self,
        pid: int,
        thread_groups: Dict[str, List[int]] = None,
        output_file_dir: str = ".",
        perf_path: str = "/usr/bin/perf",
        recording_frequency: int = 99,
        events: List[str] = None,
        metric_groups: List[str] = None,
        sampling_interval_ms: int = None,
        per_thread: bool = True,
    ):
        self._output_file_path = f"{output_file_dir}/perf.data"
        self._pid = pid
        self._thread_groups = thread_groups
        self._thread_stats: Dict = {}
        self._perf_stats_threads: Dict[str, subprocess.Popen] = {}
        self._perf_stats_start_times: Dict[str, float] = {}
        self._per_thread = per_thread

        self._perf_path = perf_path
        if not (path.exists(self._perf_path)):
            logging.error(
                f"Supplied perf path: {self._perf_path} doesn't exist"
            )

        self._events_args: List[str] = []
        if events:
            self._events_args = ["-e", ",".join(events)]

        self._metrics_groups_args: List[str] = []
        if metric_groups:
            self._metrics_groups_args = [
                "-M",
                ",".join(",".join(metric_groups)),
            ]

        self._sampling_interval_args: List[str] = []
        if sampling_interval_ms:
            self._sampling_interval_args = [
                "-I",
                str(int(sampling_interval_ms)),
            ]

        self._record_cmd = (
            [
                self._perf_path,
                "record",
                "-F",
                str(recording_frequency),
                "-o",
                self._output_file_path,
                "-p",
                str(self._pid),
            ]
            + self._events_args
            + self._metrics_groups_args
            + self._sampling_interval_args
        )

        self._stats_cmd = (
            [
                self._perf_path,
                "stat",
                "--per-thread" if self._per_thread else "",
                "-x",
                ",",
                "-p",
                str(self._pid),
            ]
            + self._events_args
            + self._metrics_groups_args
            + self._sampling_interval_args
        )

    def start_stats_profile(self):
        self._perf_stats_process = subprocess.Popen(
            self._stats_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        self._perf_stats_process_start_time = time()

    def _get_clean_perf_stats_df(
        self,
        headers: List[str],
        perf_stats_output_csv: StringIO,
        start_time: float,
    ):
        df = pd.read_csv(
            perf_stats_output_csv,
            sep=",",
            names=headers,
        )
        # Remove non counted intervals from df
        df = df[df["counts"] != "<not counted>"]
        # Convert result to numeric values
        df["counts"] = pd.to_numeric(df["counts"])
        df["tid"] = pd.to_numeric(df["tid"].str.split("-").str[-1])
        df["elapsed"] = df["elapsed"] + start_time
        return df

    def _get_perf_stat_group_command(self, thread_ids: List[int]):
        return (
            [
                self._perf_path,
                "stat",
                "-x",
                ",",
                "--per-thread" if self._per_thread else "",
                "-t",
                ",".join([str(thread_id) for thread_id in thread_ids]),
            ]
            + self._events_args
            + self._metrics_groups_args
            + self._sampling_interval_args
        )

    def start_stats_profile_threads(self):
        if self._thread_groups:
            for group_name, thread_ids in self._thread_groups.items():
                self._perf_stats_threads[group_name] = subprocess.Popen(
                    self._get_perf_stat_group_command(thread_ids),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                )
                self._perf_stats_start_times[group_name] = time()
        else:
            logging.error("No thread groups to profile")

    def profile_new_thread_group(self, name: str, thread_ids: List[int]):
        if not (self._thread_groups):
            self._thread_groups = {name: thread_ids}

        self._perf_stats_threads[name] = subprocess.Popen(
            self._get_perf_stat_group_command(thread_ids),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        self._perf_stats_start_times[name] = time()

    def stop_stats_profile_threads(
        self,
    ) -> pd.DataFrame:
        headers = [
            "elapsed",
            "tid",
            "counts",
            "units",
            "event",
            "counter_runtime",
            "percent_time_counter",
            "additional",
            "additional_units",
        ]
        dfs = []
        if self._thread_groups:
            for group_name, process in self._perf_stats_threads.items():
                return_code = process.poll()
                if not (return_code):
                    process.send_signal(signal.SIGINT)
                    try:
                        (
                            _,
                            errs,
                        ) = process.communicate(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        (
                            _,
                            errs,
                        ) = process.communicate()
                    perf_stats_output_csv = StringIO(errs)
                    df = self._get_clean_perf_stats_df(
                        headers,
                        perf_stats_output_csv,
                        self._perf_stats_start_times[group_name],
                    )
                    df["group_name"] = group_name
                    dfs.append(df)
                elif return_code == 129:
                    raise ValueError("Perf exited without recording stats")
                else:
                    (
                        _,
                        errs,
                    ) = process.communicate()
                    raise ValueError(
                        f"Perf exited with code {return_code}. Output: {errs}"
                    )

            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def stop_stats_profile(self) -> pd.DataFrame:
        headers = [
            "elapsed",
            "counts",
            "units",
            "event",
            "counter_runtime",
            "percent_time_counter",
            "additional",
            "additional_units",
        ]
        if self._per_thread:
            headers.insert(1, "tid")
        return_code = self._perf_stats_process.poll()
        if not (return_code):
            # Send sigint to record process
            self._perf_stats_process.send_signal(signal.SIGINT)
        elif return_code == 129:
            raise ValueError("Perf exited without recording stats")

        try:
            outs, errs = self._perf_stats_process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self._perf_stats_process.kill()
            outs, errs = self._perf_stats_process.communicate()

        perf_stats_output_csv = StringIO(errs)
        return self._get_clean_perf_stats_df(
            headers, perf_stats_output_csv, self._perf_stats_process_start_time
        )
