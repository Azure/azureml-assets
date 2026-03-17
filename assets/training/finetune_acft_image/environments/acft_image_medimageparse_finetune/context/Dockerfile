FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9

# protobuf is a transitive dep of mlflow-skinny/onnx; parents require >=3.12.0, cannot force 6.33.5
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 setuptools>=82.0.1 protobuf==6.33.5 cryptography==46.0.5 pillow==12.1.1
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# vulnerability in base conda env
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip>=26.0.1 wheel>=0.46.2 setuptools>=82.0.1 cryptography==46.0.5