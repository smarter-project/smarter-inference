import argparse
import os
import pathlib

import pandas as pd
from model_analyzer.triton.model.model_config import ModelConfig


def get_tflite_accel_options(row, config_base_dir: str):
    model_config_dict = ModelConfig.create_from_file(
        os.path.join(config_base_dir, row["Model Config Path"])
    ).get_config()

    try:
        cpu_exec_accel = model_config_dict["optimization"][
            "execution_accelerators"
        ]["cpu_execution_accelerator"][0]["name"]
    except KeyError:
        cpu_exec_accel = "default"

    try:
        tflite_thr = model_config_dict["optimization"][
            "execution_accelerators"
        ]["cpu_execution_accelerator"][0]["parameters"]["num_threads"]
    except KeyError:
        tflite_thr = "None"

    try:
        reduce_fp32_to_fp16 = model_config_dict["optimization"][
            "execution_accelerators"
        ]["cpu_execution_accelerator"][0]["parameters"]["reduce_fp32_to_fp16"]
    except KeyError:
        reduce_fp32_to_fp16 = "None"

    try:
        reduce_fp32_to_bf16 = model_config_dict["optimization"][
            "execution_accelerators"
        ]["cpu_execution_accelerator"][0]["parameters"]["reduce_fp32_to_bf16"]
    except KeyError:
        reduce_fp32_to_bf16 = "None"

    try:
        fast_math_enabled = model_config_dict["optimization"][
            "execution_accelerators"
        ]["cpu_execution_accelerator"][0]["parameters"]["fast_math_enabled"]
    except KeyError:
        fast_math_enabled = "None"

    return (
        cpu_exec_accel,
        tflite_thr,
        reduce_fp32_to_fp16,
        reduce_fp32_to_bf16,
        fast_math_enabled,
    )


def update_csv(metrics_file_path: str, config_base_dir: str, in_place: bool):
    df = pd.read_csv(metrics_file_path)

    (
        df["cpu_accelerator"],
        df["cpu_accelerator_num_threads"],
        df["armnn_fp32_to_fp16"],
        df["armnn_fp32_to_bf16"],
        df["armnn_fast_math_enabled"],
    ) = zip(
        *df.apply(
            get_tflite_accel_options, axis=1, config_base_dir=config_base_dir
        )
    )

    pathlib_file = pathlib.Path(metrics_file_path)
    file_name = (
        metrics_file_path
        if in_place
        else pathlib_file.stem + "_new" + pathlib_file.suffix
    )
    df.to_csv(file_name, index=False)
    print(f"Output written to: {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--metrics-file-path",
        type=str,
        required=True,
        help="path to target metrics file",
    )
    parser.add_argument(
        "-r",
        "--model-config-repo",
        type=str,
        required=True,
        help="directory of triton model output configs",
    )
    parser.add_argument(
        "-i",
        "--in-place",
        required=False,
        default=False,
        action="store_true",
        help="Overwrite existing csv",
    )
    args, unknown = parser.parse_known_args()

    update_csv(args.metrics_file_path, args.model_config_repo, args.in_place)
