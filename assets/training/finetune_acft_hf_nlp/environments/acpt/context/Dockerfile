FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

ENV PIP_PROGRESS_BAR=off

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
      dotnet-hostfxr-8.0 \
      dotnet-host-8.0 \
      dotnet-runtime-8.0 \
      libgcrypt20 \
      liblzma5 \
      xz-utils || true

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

# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# python-multipart: transitive dep (fastmcp → fastapi → python-multipart); fastapi pins >=0.0.18; override needed (GHSA-mj87-hwqh-73pj)
# Mako: transitive dep (mlflow → alembic → Mako); alembic has no version constraint; override needed (GHSA-v92g-xgxw-vvmm)
# python-dotenv: transitive dep (fastmcp → python-dotenv); fastmcp 3.2.4 uses >=1.1.0; override needed (GHSA-mf9w-mj56-hr94)
# onnx: azureml-acft-accelerator 0.0.89 caps onnx<=1.17.0; override needed for GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m etc.
# skops: transitive dep (mlflow → skops); mlflow 3.11.1 declares 'skops<1' (loose floor), so pip
#        resolves to 0.11.0 which has CVE-2025-54412/54413/54886
#        (GHSA-m7f4-hrc6-fwg3, GHSA-4v6w-xpmh-gfgp, GHSA-378x-6p4f-8jgm); override to >=0.13.0
# urllib3: transitive dep brought in by many parents (requests, botocore, kubernetes, etc.); the closest
#          parent `requests` declares `urllib3>=1.26,<3` (loose floor) so upgrading requests does
#          not pull urllib3>=2.7.0; direct override required for GHSA-mf9v-mfxr-j63j, GHSA-qccp-gfcp-xxvc
RUN pip install --upgrade --no-cache-dir pyasn1==0.6.3 'python-multipart>=0.0.26' 'Mako>=1.3.11' 'python-dotenv>=1.2.2' 'onnx>=1.21.0' 'skops>=0.13.0' 'urllib3>=2.7.0'

# pip is the package manager in both base and ptca conda envs. defaults has no pip>=26.1.1 package,
# while a normal conda-forge solve pulls unrelated conda updates and py-rattler. Install only the
# patched pip package so conda-meta and site-packages both report the fixed version.
RUN conda install -n base -c conda-forge --no-deps -y 'pip>=26.1.1' && \
    conda install -n ptca -c conda-forge --no-deps -y 'pip>=26.1.1'

# aiohttp and cryptography in ptca env (Python 3.10) are present in the base image; force upgrades here explicitly.
# cryptography is pinned as a transitive/base image package to fix GHSA-537c-gmf6-5ccf.
RUN conda run -n ptca python -m pip install --upgrade --no-cache-dir 'aiohttp>=3.14.0' 'cryptography>=48.0.1'

# torch is inherited from the ACPT base image and required by torchvision/torchaudio;
# upgrade the matching CUDA 12.6 PyTorch wheel set to fix GHSA-vgrw-7cvw-pwgx and GHSA-qfhq-4f3w-5fph.
RUN conda run -n ptca python -m pip install --upgrade --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.10.0+cu126' \
    'torchvision==0.25.0+cu126' \
    'torchaudio==2.10.0+cu126'

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
