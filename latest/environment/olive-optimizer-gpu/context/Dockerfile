FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:20231226.v1

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# Build python3.9 as the default python
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        software-properties-common gnupg\
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends \
        python3.9-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    python3.9 -m venv /venv

ENV PATH=/venv/bin:$PATH

# Install TensorRT
RUN v="8.4.1-1+cuda11.6" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub &&\
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v} libnvinfer-samples=${v}

RUN pip install --upgrade pip && pip install --upgrade setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir