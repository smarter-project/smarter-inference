
ARG TRITON_BASE_IMAGE=tritonserver:latest

FROM ${TRITON_BASE_IMAGE} as tritonserver_build_final

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    libb64-0d \
    libcurl4-openssl-dev \
    libre2-5 \
    git \
    dirmngr \
    libnuma-dev \
    curl \
    python3-dev \
    python3-pip \
    pkg-config \
    libcairo2-dev \
    libjpeg-dev \
    libgif-dev \
    gcc \
    cpuset \
    nginx && \
    rm -rf /var/lib/apt/lists/*

# Create a output model repository for the admission controller
RUN mkdir /opt/output_models

COPY requirements-admission-controller.txt /opt/requirements-admission-controller.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir wheel && \
    python3 -m pip install --no-cache-dir -r /opt/requirements-admission-controller.txt

# To support both arm and x86 versions, we must build model_analyzer from source
# The pip wheel for the arm sbsa model analyzer does not work
RUN git clone -b r22.05 https://github.com/triton-inference-server/model_analyzer /tmp/model_analyzer && \
    cd /tmp/model_analyzer && \
    chmod +x build_wheel.sh && \
    ./build_wheel.sh /usr/local/bin/perf_analyzer true && \
    python3 -m pip install --no-cache-dir --ignore-installed wheels/triton_model_analyzer-*.whl && \
    rm -rf /tmp/model_analyzer

COPY main.py /opt/main.py
COPY admission_controller /opt/admission_controller

WORKDIR /opt
ENTRYPOINT [ "python3", "main.py" ]
