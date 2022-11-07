# Admission Controller for Triton Inference Server

## Installation
To install the admission controller, you can either build against the host machine or use docker

To initialize the repo with our test images you can run from the root of this repo:
```bash
cd admission_controller/tests
wget https://images.getsmarter.io/ml-models/admission_controller_triton_model_repo.tar.gz
tar -xvzf admission_controller_triton_model_repo.tar.gz
```

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
To build your docker image you can simply run the following:
```bash
cd <admission_controller_directory>
docker build -t admission_controller .
```
and if building for jetson devices you can run:
```bash
cd <admission_controller_directory>
docker build -f Dockerfile.jetson -t admission_controller .
```


If you wish to build against a custom version of triton you can build triton from scratch:
```bash
cd ~
git clone -b r22.05 https://github.com/triton-inference-server/server.git
cd server
./build.py --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=tensorflow1 --backend=armnn_tflite:dev --extra-backend-cmake-arg=tensorflow1:TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS=ON
```
This command will generate a `tritonserver:latest` image, which will be used as the base image in the next step

Now build the admission controller, specifying your custom triton image:
```bash
cd <admission_controller_directory>
docker build --build-arg TRITON_BASE_IMAGE=tritonserver:latest -t admission_controller .
```

### Host build
To build against the host, you need to install triton yourself, which is easiest to do via a host build. On an Ubuntu 20.04 machine, you can run (replace 22.05 with the NGC container release version of choice):
```bash
git clone -b r22.05 https://github.com/triton-inference-server/server
cd server
./build.py --cmake-dir=$HOME/code/server/build --build-dir=$HOME/citritonbuild --no-container-build --enable-logging --enable-stats --enable-tracing --enable-metrics --endpoint=http --endpoint=grpc --backend=tensorflow1 --backend=armnn_tflite:dev --extra-backend-cmake-arg=tensorflow1:TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS=ON --upstream-container-version=22.05
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
docker build --build-arg TRITON_BASE_IMAGE=tritonserver:latest -t model-analyzer -f Dockerfile.custom_triton .
```

For Jetson builds you can build the Docker image for triton and the admission controller and use this for the custom model analyzer build:
```bash
docker build --build-arg TRITON_BASE_IMAGE=admission_controller -t model-analyzer -f Dockerfile.custom_triton .
```

**Before you run the profiler you will want to lock your clock freqencies by setting your cpu scaling governor to performance (max freq) or powersave (min freq).**

To profile the models in this repo:
```bash
docker run -it --rm \
     -v $HOME/code/admission_controller/admission_controller/tests:/model-test \
     --net=host -v $HOME/saved_checkpoints:/opt/triton-model-analyzer/checkpoints model-analyzer

model-analyzer profile --model-repository /model-test/triton_model_repo/ --config /model-test/triton_model_repo/profile_config_<tf/tflite>_<machine_name>.yml --collect-cpu-metrics --override-output-model-repository
```

Finally to export the data into this repository, from the root of this repo, run the script:
```bash
./export_model_data.sh -c <container id of model analyzer> -p <platform name>
```
where platform name is an arbitrary nickname of the machine the data was gathered

The output of these instructions will leave a csv file for use with the admission controller

### Admission Controller Runtime Options
Admission Controller is configurable using environment variables. The following options are available:
| Variable | Values | Default | Details |
| ---      | ---    | ---     | ---     |
| `SHIELD`   | Any or unset   | unset | If set use cpu shield assuming it has already been setup on the host |
| `GPUS`     | List of gpu ids  | unset | Gpu ids. On jetson only option is 0  |
| `SCHED`     | `NICE`, `DEADLINE`, `RR`, or unset | unset | Linux scheduling class to use, if unset inference threads run with default priority  |
| `MULTI_SERVER` | Any or unset | unset | If set, one triton server process will be used per managed model  |
| `TRITON_INSTALL_PATH` | Filepath or unset | `/opt/tritonserver` | Path to tritonserver binary |
| `OUTPUT_REPO_PATH` | Filepath or unset | `/opt/output_models` | Location to write output model repo for triton to use |
| `CLIENT_MAX_RETRIES` | Integer or unset | 20 | Max retries for a triton client used by admission controller when managing triton |
| `TRITON_URL` | URL or unset | `localhost:8000` | Url where admission controller can access triton |
| `PRIORITY_ASSIGNMENT` | `SLACK` or unset  | unset | If `SLACK`, when using `RR` or `NICE` scheduling class: assign highest priority to inference with tightest deadlines, else assign highest priority to lowest request latency |
| `SCALING_FACTOR` | Float or unset | 1.0 | Scale profiled inference latency data by scaling factor to account for clock freq changes |
| `NO_ENFORCE` | Any or unset | unset | If set admission control is not enforced by the system |
| `NGINX_LOGGING` | Any or unset | unset | If set enable nginx logs to be written |
| `NGINX_CONFIG_PATH` | Filepath or unset | `/tmp/nginx_config.txt` | File to write nginx logs |
| `NO_NGINX_RATELIMIT` | Any or unset | unset | If set nginx rate-limiting not used |

### Run via Docker
To run the docker image simply run:
```bash
docker run --rm -it -p 2520-2522:2520-2522 admission_controller
```

### Run on host
To run on the host you must pass some environment variables to point to your triton installation and desired output model repository. As an example you may run:
```bash
python3 -m pip install -r requirements-admission-controller.txt
TRITON_INSTALL_PATH=$HOME/tritonserver_install OUTPUT_MODEL_REPO_PATH=/tmp/output_model_repository python3 main.py
```

By default the admission control api is available on `0.0.0.0:2520`
The Nginx server acts as the gateway to access the triton server, and listens on `0.0.0.0:2521` for http requests to Triton, and `0.0.0.0:2522` for gRPC requests.

### Run test suite
To run the test suite to measure repeatable performance for a fixed set of tests we use pytest. These tests rely on docker to build an run the admission controller image. 

From the root of this repository run the following commands:
```bash
python3 -m pip install -r test_deps.txt
pytest --plot -vv --min-capture-duration 30 --max-capture-duration 200
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
