FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202602.2

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
    libpam-runtime && \
    apt-get install -y sudo=1.9.9-1ubuntu2.5 && \
    apt-mark hold sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

# try updating pip for base and ptca env using conda
RUN conda install pip -n base -y || true
RUN conda install pip -n ptca -y || true

# This ensures necessary torch dependencies are already installed
RUN pip install --no-cache-dir torch==2.8.0  torchaudio==2.8.0 torchvision==0.23.0
RUN pip install --no-cache-dir \ 
    'azureml-automl-dnn-nlp==1.62.0' \
    'azureml-defaults==1.62.0'

# onnx and onnxruntime-training installation
RUN pip uninstall -y onnxruntime
RUN pip install onnx==1.17.0
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.18.0

# Install numpy
RUN pip install --upgrade "numpy==1.23.5"
RUN /opt/conda/bin/pip install --upgrade "numpy==1.23.5" || true

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
RUN /opt/conda/bin/pip install --upgrade 'urllib3>=2.6.3' || true
RUN pip install \
    'pyarrow==14.0.1' \
    'aiohttp>=3.13.3' \
    'idna==3.7' \
    'requests==2.32.5' \
    'urllib3>=2.6.3' \
    'mlflow-skinny==2.15.1' \
    'filelock>=3.20.1'

# Patch for Pillow vulnerability
RUN pip install --upgrade 'pillow==12.0.0'
RUN /opt/conda/bin/pip install --upgrade 'pillow==12.0.0' || true

# Fix security vulnerabilities
RUN pip install --upgrade 'bokeh==3.8.2' 'PyNaCl==1.6.2'

RUN pip list
