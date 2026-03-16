FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root:root

# Update system package index and upgrade Python 3.10 packages to required versions
RUN apt-get update && \
    apt-get install -y --only-upgrade \
    libpython3.10-stdlib \
    python3.10 \
    libpython3.10-minimal \
    python3.10-minimal \
    libpam0g \
    libpam-modules-bin \
    libpam-modules \
    libc-bin \
    libc-dev-bin \
    binutils \
    binutils-common \
    binutils-x86-64-linux-gnu \
    libbinutils \
    libctf0 \
    libctf-nobfd0 \
    libxml2 \
    dpkg \
    dpkg-dev \
    dotnet-hostfxr-8.0 \
    dotnet-host-8.0 \
    dotnet-runtime-8.0 \
    linux-libc-dev \
    linux-headers-generic \
    locales \
    openssl \
    libssl-dev \
    libssh-4 \
    libpam-runtime \
    libexpat1 \
    libpng16-16 \
    libgnutls30 \
    libasound2 \
    libasound2-data \
    curl \
    libcurl3-gnutls \
    libcurl4 && \
    apt-get install -y sudo=1.9.9-1ubuntu2.5 && \
    apt-mark hold sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

# try updating pip for base and ptca env using conda
RUN conda install pip -n base -y
RUN conda install pip -n ptca -y

RUN pip install --no-cache-dir \ 
    'azureml-automl-dnn-nlp=={{latest-pypi-version}}' \
    'azureml-defaults=={{latest-pypi-version}}'

# onnx and onnxruntime-training installation
RUN pip uninstall -y onnxruntime
RUN pip install onnx==1.17.0
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.18.0

# torch-ort installation
RUN TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
RUN pip install torch-ort==1.18.0 && TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m torch_ort.configure
RUN pip uninstall -y onnxruntime

RUN pip install \ 
    transformers==4.53.0 \
    optimum==1.23.3 \
    accelerate==1.12.0 \
    deepspeed~=0.15.1

# Address vulnerabilities
# Patch for Pillow vulnerability : Direct dep (used by bokeh, torchvision) from base image
RUN pip install --upgrade 'pillow>=12.1.1'

# Fix security vulnerabilities (ptca env)
RUN pip install --upgrade 'wheel>=0.46.2' 'cryptography>=46.0.5' 'setuptools>=82.0.1' 'distributed>=2026.1.0' 'filelock>=3.20.3' 'bokeh>=3.8.2' 'protobuf>=6.33.5'

# Fix security vulnerabilities (conda base env)
RUN /opt/conda/bin/pip install --upgrade 'cryptography>=46.0.5' 'wheel>=0.46.2' 'setuptools>=82.0.1'

