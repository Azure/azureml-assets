# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch220:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Pinning to 1.16.0 for now to avoid swinv2 segfault issue with ORT.
RUN pip uninstall onnxruntime-training -y && pip install onnxruntime-training==1.16.0
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install

# dummy number to change when needing to force rebuild without changing the definition: 1