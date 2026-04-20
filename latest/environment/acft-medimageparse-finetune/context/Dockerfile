FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.2

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
# mlflow-skinny 2.13.2 has CVEs; upgrade after requirements install
# azureml-mlflow pins mlflow-skinny<=3.5.0, so must be upgraded separately to avoid resolution conflict
RUN pip install --no-cache-dir mlflow-skinny==3.10.1
# aiohttp: transitive dep of azure-core; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: transitive dep of onnxruntime; parent uses onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# nltk: GHSA-gfwx-w7gr-fvh7; >=3.9.4 required
# pydicom: GHSA-v856-2rf8-9f28; requirements.txt pins ~=2.4.0 allowing vulnerable 2.4.4; override to >=2.4.5
# requests: transitive dep of azure-core/mlflow; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 setuptools>=82.0.1 protobuf==6.33.5 cryptography==46.0.7 pillow==12.1.1 'requests>=2.33.0' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'nltk>=3.9.4' 'pydicom>=2.4.5'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y pip>=26.0.1 wheel>=0.46.2
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip>=26.0.1 wheel>=0.46.2 setuptools>=82.0.1 cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0'