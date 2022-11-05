# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import select
import sys
import threading
from math import ceil
from re import L
from time import sleep, time
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

run_inference_loop = threading.Event()


def get_random_inputs(model_config_dict, batch_size, client_module):
    inputs = []
    for i, input in enumerate(model_config_dict["input"]):
        input_dims = [int(i) for i in input["dims"]]

        if batch_size:
            input_dims.insert(0, batch_size)

        dtype_str = input["data_type"].split("_")[1]
        inputs.append(
            client_module.InferInput(input["name"], input_dims, dtype_str)
        )

        # Initialize input data using random values
        random_array = np.random.randn(*input_dims).astype(
            triton_to_np_dtype(dtype_str), copy=False
        )
        inputs[i].set_data_from_numpy(random_array)
    return inputs


def infer(
    client_type: str,
    endpoint_uuid: str,
    client_id: str,
    model_name: str,
    model_config_dict_str: str,
    triton_url: str,
    period: float,
    batch_size: Optional[int],
    retries: int = 5,
    update_interval: int = 10,
    regular_sched=False,
):
    """
    Sends inference requests to Triton at approximately the period (seconds) argument passed.
    If deadline missed will send back message to the parent process to notfiy
    """

    model_config_dict = json.loads(model_config_dict_str)

    failures = 0
    inference_count = 0
    update_samples = int(update_interval / period)

    # If period longer than update interval, just update every inference
    if update_samples == 0:
        update_samples = 1

    if client_type == "http":
        client_module = httpclient
    elif client_type == "grpc":
        client_module = grpcclient
    else:
        print("Unknown client module. Select from either http or grpc")
        sys.exit(1)

    triton_client = client_module.InferenceServerClient(triton_url)

    # Check that model is ready before beginning inference loop
    retry_count = 0
    while not (triton_client.is_model_ready(endpoint_uuid)):
        if retry_count > retries:
            msg = f"Client {model_name}-{client_id} model did not become ready"
            print(msg)
            sys.exit(1)
        retry_count += 1
        sleep(1)

    while True:
        # Wait for inference loop signal to be set
        run_inference_loop.wait()

        finish = timer()
        start = timer()

        # Latencies holds timestamps for the start time stamp of the inference, followed by
        # the measured inference latency
        latencies = []
        client_overhead_start = None

        # Wait for start signal to begin
        while True:
            inputs = get_random_inputs(
                model_config_dict, batch_size, client_module
            )
            if client_overhead_start:
                finish += timer() - client_overhead_start
            timestamp = timer()
            if finish < timestamp:
                # If we missed a deadline, finish will be less than current time
                # In that case, we need to advance finish accordingly
                if regular_sched:
                    finish += (ceil((timestamp - finish) / period)) * period
                else:
                    # Else don't advance finish, and send next request immediately after missing the deadline
                    finish = timestamp

            sleep(finish - timestamp)
            finish += period

            start = timer()
            latencies.append(time())
            try:
                results = triton_client.infer(
                    endpoint_uuid, inputs, outputs=None, headers=None
                )
            except Exception as e:
                # Ignore failures because of Nginx not being ready for now
                if "Request for unknown model" in str(
                    e
                ) or "Parse error at offset 0" in str(e):
                    print(
                        f"Client {model_name}-{client_id} Inference Issue!"
                        f" error: {e}"
                    )
                    continue
                else:
                    print(
                        f"Client {model_name}-{client_id} Inference Failed!"
                        f" error: {e}"
                    )
                    failures += 1
                    # If we have failed 10 times we exit the loop as something is likely wrong
                    if failures > 10:
                        print(
                            f"Client {model_name}-{client_id} Exiting for too"
                            " many failures"
                        )
                        sys.exit(1)
            inf_latency = timer() - start
            client_overhead_start = timer()
            latencies.append(inf_latency)
            inference_count += 1

            if inference_count % update_samples == 0:
                msg = f"latencies,{','.join(map(str, latencies))}"
                latencies = []
                print(msg)

            if not (run_inference_loop.is_set()):
                break


def control():
    try:
        ep = select.epoll()
        ep.register(
            sys.stdin.fileno(),
            select.EPOLLIN | select.EPOLLHUP,
        )
        while True:
            evts = ep.poll()
            for fd, evt in evts:
                if evt & select.EPOLLIN:
                    cmd = sys.stdin.readline().rstrip()
                    print(cmd)
                    if cmd == "start":
                        run_inference_loop.set()
                        print("started")
                    elif cmd == "halt":
                        run_inference_loop.clear()
                        print("halted")
                if evt & select.EPOLLHUP:
                    for line in sys.stdin:
                        print(line)
                    ep.unregister(fd)

    except Exception as e:
        print(e)

    finally:
        ep.close()


if __name__ == "__main__":

    inf_thread = threading.Thread(
        target=infer,
        kwargs={
            "client_type": sys.argv[1],
            "endpoint_uuid": sys.argv[2],
            "client_id": sys.argv[3],
            "model_name": sys.argv[4],
            "model_config_dict_str": sys.argv[5],
            "triton_url": sys.argv[6],
            "period": float(sys.argv[7]),
            "batch_size": int(sys.argv[8]),
            "update_interval": int(sys.argv[9]),
            "regular_sched": sys.argv[10] == "True",
        },
        daemon=True,
    )

    control_thread = threading.Thread(target=control, daemon=True)

    control_thread.start()
    inf_thread.start()
    inf_thread.join()
    control_thread.join()
