FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202603.1

USER root:root

# Update system package index and upgrade Python 3.10 packages to required versions
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

# try updating pip for base and ptca env using conda
RUN conda install pip -n base -y
RUN conda install pip -n ptca -y

RUN pip install --no-cache-dir \ 
    'azureml-automl-dnn-nlp==1.62.0' \
    'azureml-defaults==1.62.0'

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
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
RUN pip install --upgrade 'wheel>=0.46.2' 'cryptography>=46.0.5' 'setuptools>=82.0.1' 'distributed>=2026.1.0' 'filelock>=3.20.3' 'bokeh>=3.8.2' 'protobuf>=6.33.5'

# Fix security vulnerabilities (conda base env)
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN /opt/conda/bin/pip install --upgrade 'cryptography>=46.0.5' 'wheel>=0.46.2' 'setuptools>=82.0.1' 'PyJWT>=2.12.0'

