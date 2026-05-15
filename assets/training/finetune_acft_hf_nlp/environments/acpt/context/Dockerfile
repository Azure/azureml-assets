#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && apt-get -y upgrade

COPY requirements.txt .
# The below 2 files are required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
COPY finetune_run.py /azureml/finetune/run.py

# mpi4py 3.x uses distutils APIs removed in setuptools>=81; upgrade to 4.x which is compatible
RUN pip install mpi4py==4.1.1 --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
# Verify QLoRA dependencies in the ptca runtime env during image build.
RUN /opt/conda/envs/ptca/bin/python -c "import bitsandbytes as bnb; print(bnb.__version__)" && \
    /opt/conda/envs/ptca/bin/python -c "from transformers import BitsAndBytesConfig; print(BitsAndBytesConfig(load_in_4bit=True))"

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
# skops: transitive dep resolved to 0.11.0; override needed for GHSA-m7f4-hrc6-fwg3, GHSA-4v6w-xpmh-gfgp, GHSA-378x-6p4f-8jgm
RUN pip install --upgrade --no-cache-dir pyasn1==0.6.3 'python-multipart>=0.0.26' 'Mako>=1.3.11' 'python-dotenv>=1.2.2' 'onnx>=1.21.0' skops==0.13.0

# python-dotenv 1.2.1 in base conda env (python3.13) from ACPT base image needs upgrade (GHSA-mf9w-mj56-hr94)
# parents in base env are anaconda-auth (no version pin) and pydantic-settings (>=0.21.0); both use loose floors
# so upgrading parents does not pull in 1.2.2; direct override required until base image refreshes
RUN conda run -n base python -m pip install --upgrade --no-cache-dir 'python-dotenv>=1.2.2'

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/

