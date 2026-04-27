#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.2

USER root

RUN apt-get update && apt-get -y upgrade

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

# mpi4py 3.x uses distutils APIs removed in setuptools>=81; upgrade to 4.x which is compatible
RUN pip install mpi4py==4.1.1 --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install mlflow==3.11.1
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN MAX_JOBS=$(nproc) pip install --no-cache-dir --upgrade flash-attn==2.8.3 --no-build-isolation
RUN pip install nltk==3.9.4 # Pinning to fix the unsafe deserialization vulnerability

# vulnerabilities, cannot be added to requirements.txt as it causes pip dependency resolver to break
# fastmcp: GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; >=3.2.0 required
RUN pip install --upgrade --no-cache-dir 'fastmcp>=3.2.0'

# protobuf is required by onnxruntime, mlflow
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# pyans1 is required by azureml-acft-accelerator, mlflow and both dont pin the libs hence upgrade
# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# aiohttp: transitive dep of ray/vllm/azure-core; parents use loose floors (>=3.10); override needed (GHSA-mwh4-6h8g-pg8w etc.)
# onnx: onnxruntime/azureml-acft-accelerator require onnx>=1.16.0; override needed (GHSA-p433-9wv8-28xj etc.)
# requests: transitive dep of azure-core/mlflow/transformers; parents use loose floors (GHSA-gc5v-m9x4-r6x2)
# pillow: GHSA-whj4-6x5x-4v2j; >=12.2.0 required
# pytest: pre-installed in ACPT base image; no parent package to upgrade (GHSA-6w46-j5rx-g56g)
# transformers: final override to ensure >=5.0.0 after azureml-* installs (GHSA-69w3-r845-3855)
# python-multipart: transitive dep (fastmcp → fastapi → python-multipart); parent uses loose floor; override needed (GHSA-mj87-hwqh-73pj)
# Mako: transitive dep (mlflow → alembic → Mako); parent uses loose floor; override needed (GHSA-v92g-xgxw-vvmm)
RUN pip install --upgrade pip>=26.0 wheel>=0.46.2 protobuf==6.33.5 pyasn1==0.6.3 cryptography==46.0.7 pillow==12.2.0 'python-multipart>=0.0.26' 'aiohttp>=3.13.4' 'onnx>=1.21.0' 'requests>=2.33.0' 'pytest>=9.0.3' 'transformers>=5.0.0' 'Mako>=1.3.11'
# pip install updates the binary but conda-meta still references old versions; conda install syncs both
RUN conda install -n ptca -y wheel>=0.46.2 pip>=26.0.1
# vulnerability in base conda env
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base python -m pip install --upgrade pip>=26.0 wheel>=0.46.2 setuptools>=82.0.0 cryptography==46.0.7 'PyJWT>=2.12.0' 'aiohttp>=3.13.4' 'requests>=2.33.0'

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/

