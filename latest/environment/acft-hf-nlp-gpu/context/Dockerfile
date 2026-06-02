#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.1

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

# pyasn1 is a transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
# parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# python-multipart: transitive dep (fastmcp → fastapi → python-multipart); fastapi pins >=0.0.18; override needed (GHSA-mj87-hwqh-73pj)
# Mako: transitive dep (mlflow → alembic → Mako); alembic has no version constraint; override needed (GHSA-v92g-xgxw-vvmm)
# python-dotenv: transitive dep (fastmcp → python-dotenv); fastmcp 3.2.4 uses >=1.1.0; override needed (GHSA-mf9w-mj56-hr94)
# onnx: azureml-acft-accelerator 0.0.89 caps onnx<=1.17.0; override needed for GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m etc.
# skops: transitive dep (mlflow → skops); mlflow 3.12.0 declares 'skops<1' (loose floor), so pip
#        resolves to 0.11.0 which has CVE-2025-54412/54413/54886
#        (GHSA-m7f4-hrc6-fwg3, GHSA-4v6w-xpmh-gfgp, GHSA-378x-6p4f-8jgm); override to >=0.13.0
# urllib3: transitive dep brought in by many parents (requests, botocore, kubernetes, etc.); the closest
#          parent `requests` 2.34.2 declares `urllib3>=1.26,<3` (loose floor) so upgrading requests does
#          not pull urllib3>=2.7.0; direct override required for GHSA-mf9v-mfxr-j63j, GHSA-qccp-gfcp-xxvc
RUN pip install --upgrade --no-cache-dir pyasn1==0.6.3 'python-multipart>=0.0.26' 'Mako>=1.3.11' 'python-dotenv>=1.2.2' 'onnx>=1.21.0' 'skops>=0.13.0' 'urllib3>=2.7.0'

# Base conda env (Python 3.13) packages scanned from /opt/conda need direct pip overrides.
# Dependency inspection shows parents only use loose ranges:
# - urllib3: requests 2.33.1 declares urllib3<3,>=1.26.
# - idna: requests 2.33.1, httpx 0.28.1, anyio 4.12.1 and yarl 1.23.0 accept vulnerable 3.11.
# - click: anaconda-cli-base 0.8.2, typer 0.25.1 and extras from httpx/python-dotenv/cryptography accept 8.2.1.
# python-dotenv has no installed parent in base; it is a vulnerable base package itself. Upgrading parents
# does not force patched child versions, and conda updates would introduce py-rattler, so use pip in base.
RUN conda run -n base python -m pip install --upgrade --no-cache-dir 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3'

# pip is the package manager in both base and ptca conda envs. defaults has no pip>=26.1.1 package,
# while a normal conda-forge solve pulls unrelated conda updates and py-rattler. Install only the
# patched pip package so conda-meta and site-packages both report the fixed version.
RUN conda install -n base -c conda-forge --no-deps -y 'pip>=26.1.1' && \
    conda install -n ptca -c conda-forge --no-deps -y 'pip>=26.1.1'

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/

