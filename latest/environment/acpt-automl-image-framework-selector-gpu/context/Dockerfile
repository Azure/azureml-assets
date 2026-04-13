# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.1

RUN apt-get -y update && apt-get -y upgrade
# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# upgrade pip, wheel, protobuf vulnerabilities in ptca env
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# setuptools vendors jaraco.context internally; >=82.0.1 bundles the patched version (GHSA-58pv-8j8x-9vj2)
# protobuf parent is mlflow (already latest) which accepts protobuf>=3.12.0, cannot force 6.33.5 via parent
# aiohttp: transitive dep of azure-core; parents use loose floors (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: transitive dep of onnxruntime; parent uses onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# requests: transitive dep of azure-core/mlflow; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
RUN pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 cryptography==46.0.7 'setuptools>=82.0.1' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'requests>=2.33.0'

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --no-cache-dir --upgrade pip==26.0 wheel==0.46.2 'setuptools>=82.0.1' cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -y -n ptca pip>=26.0.1 wheel>=0.46.2
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
