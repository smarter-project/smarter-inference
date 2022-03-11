# Admission Controller for Triton Inference Server

## Installation
To install the admission controller, you can either build against the host machine or use docker

### Docker Configuration
The admission controller realtime features only work on systems which use cgroup v1. You can check on your system by running:
```bash
stat -fc %T /sys/fs/cgroup/
```
If the output is `tmpfs` you are using cgroup v1, if it is `cgroup2fs` you are using cgroup v2.

On your host machine, configure docker to use the systemd cgroup driver, such that the admission controller can modify cpuset configurations in cgroupfs for realtime tasks by modifying `/usr/lib/systemd/system/docker.service`, adding 
```bash
--exec-opt native.cgroupdriver=systemd
```
to the `ExecStart` command in the systemd service.

You can then restart docker and remove the cpuset it created (if it exists) by running:
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo rmdir /sys/fs/cgroup/cpuset/docker 
```
### Docker build
To build your docker image you must first build triton from scratch, then build the admission controller.

To build the tritonserver from scratch:
```bash
cd ~
git clone -b r22.03 https://github.com/triton-inference-server/server.git
cd server
./build.py --cmake-dir=$(pwd)/build --build-dir=/tmp/citritonbuild --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=tensorflow1 --backend=armnn_tflite:main --extra-backend-cmake-arg=tensorflow1:TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS=ON
```

Now build the admission controller:
```bash
cd <admission_controller_directory>
docker build -t admission_controller .
```

### Host build
To build against the host, you need to install triton yourself, which is easiest to do via a host build. On an Ubuntu 20.04 machine, you can run (replace 22.02 with the NGC container release version of choice):
```bash
git clone -b r22.03 https://github.com/triton-inference-server/server
cd server
./build.py --cmake-dir=$HOME/code/server/build --build-dir=$HOME/citritonbuild --no-container-build --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=tensorflow1 --backend=armnn_tflite:main --extra-backend-cmake-arg=tensorflow1:TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS=ON --upstream-container-version=22.03
```

## Usage
To run the image you can either use the host or docker build above:

### Collect Profiling Data for Models
In order to collect the offline profiling data for each of your models of interest you must use a custom built verison of the Nvidia model analyzer which collects CPU utilization metrics for each model. To get this image run:
```bash
cd ~
git clone -b cpu_util https://gitlab.com/jishminor/model_analyzer.git
cd model_analyzer
docker build --build-arg TARGETARCH=amd64 -t model-analyzer .
```

If you wish to build a version of the Model Analyzer with a custom triton you can run the following after building triton using the `build.py` convenience script found [here](https://github.com/triton-inference-server/server/blob/main/build.py): 
```bash
docker build --build-arg TRITON_BASE_IMAGE=tritonserver:latest -t model-analyzer-custom -f Dockerfile.custom_triton .
```

For Jetson builds you can build the Docker image for triton and the admission controller and use this for the custom model analyzer build:
```bash
docker build --build-arg TRITON_BASE_IMAGE=admission_controller -t model-analyzer-custom -f Dockerfile.custom_triton .
```

For tensorflow models in this repo:
```bash
docker run -it --rm \
     -v $HOME/code/admission_controller/admission_controller/tests:/model-test \
     --net=host -v $HOME/saved_checkpoints:/opt/triton-model-analyzer/checkpoints model-analyzer

model-analyzer profile --model-repository /model-test/triton_model_repo/ --config /model-test/triton_model_repo/profile_config_tf_<machine_name>.yml --collect-cpu-metrics --override-output-model-repository
```

For tensorflow lite models in this repo:
```bash
docker run -it --rm \
     -v $HOME/code/admission_controller/admission_controller/tests:/model-test \
     --net=host -v $HOME/saved_checkpoints:/opt/triton-model-analyzer/checkpoints model-analyzer-custom

model-analyzer profile --model-repository /model-test/triton_model_repo/ --config /model-test/triton_model_repo/profile_config_tflite_<machine_name>.yml --collect-cpu-metrics --override-output-model-repository
```

Finally to export the data into this repository, from the root of this repo, run the script:
```bash
./export_model_data.sh -c <container id of model analyzer> -p <platform name>
```

The output of these instructions will leave a csv file for use with the admission controller

### Run via Docker
To run the docker image simply run:
```bash
docker run --rm -it -p 2520-2522:2520-2522 admission_controller
```

### Run on host
To run on the host you must pass some environment variables to point to your triton installation and desired output model repository. As an example you may run:
```bash
TRITON_INSTALL_PATH=$HOME/tritonserver_install OUTPUT_MODEL_REPO_PATH=/tmp/output_model_repository python3 main.py
```

By default the admission control api is available on `0.0.0.0:2520`
The Nginx server acts as the gateway to access the triton server, and listens on `0.0.0.0:2521` for http requests to Triton, and `0.0.0.0:2522` for gRPC requests.

### Run test suite
To run the test suite to measure repeatable performance for a fixed set of tests, first download and untar our test images then, install the test dependencies and run:
```bash
cd admission_controller/admission_controller/tests
wget https://images.getsmarter.io/ml-models/admission_controller_triton_model_repo.tar.gz
tar -xvzf admission_controller_triton_model_repo.tar.gz
python3 -m pip install -r ../../test_deps.txt
pytest --plots -vv --capture-duration 60
```

### Use CPU Shielding
To run the admission controller and make use of cpu shielding to produce more deterministic inference performance install the `cset` tool and create a shield for the admission controller to assign Triton to run on by executing:
```bash
sudo apt install cpuset
sudo cset shield --cpu 0-7 --kthread=on
```
Note you can pass any list of cpus in the above command for triton to run on this is just an example

In order to leverage the linux deadline scheduling class, after running the cset shield command above you must also run:
```bash
sudo sh -c 'echo 0 > /sys/fs/cgroup/cpuset/cpuset.sched_load_balance'
```
To disable load balancing in the root cpuset. This is necessary as without this setting, the created user and system cpusets are not considered root domains, meaning SCHED_DEADLINE scheduling class can not be applied to their tasks.

### Run perf during tests
To run perf during the tests, first you will need to ensure `perf` is on your system, and if not compile it from the linux source tree for your kernel version, then put it in `usr/bin`. Then setup a group for yourself to enable running perf without root priviledges:
```bash
sudo su
cd /usr/bin
groupadd perf_users
chgrp perf_users perf
chmod o-rwx perf
setcap cap_sys_admin,cap_sys_ptrace,cap_syslog=ep perf
```

Also enable perf to read kernel info without being root:
```bash
sudo mount -o remount,mode=755 /sys/kernel/debug/
```

If you install perf via apt, perf will be located somewhere like: `/usr/lib/linux-tools-5.15.0-40/perf`. Modify the above to account for this.

Further if you are running a kernel >= 5.8, use cap_perfmon in the capabilities as below:
```bash
sudo setcap "cap_perfmon,cap_ipc_lock,cap_sys_ptrace,cap_syslog=ep" /usr/lib/linux-tools-5.15.0-40/perf
```

Then add yourself to the group:
```bash
usermod -aG perf_users <your username>
exit
```
Also modify `/etc/sysctl.conf` file adding the line `kernel.perf_event_paranoid = -1` and run `sudo sysctl --reload`

At the end of this benchmark test being run you will see outputted graphs at `admission_controller/admission_controller/tests/plots`
