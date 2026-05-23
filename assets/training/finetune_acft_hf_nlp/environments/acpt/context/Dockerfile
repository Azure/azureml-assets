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

# Base conda env (python3.13) overrides from ACPT base image:
# - python-dotenv 1.2.1: parents anaconda-auth (no pin) and pydantic-settings (>=0.21.0) use loose
#   floors so upgrading them does not pull in 1.2.2; direct override (GHSA-mf9w-mj56-hr94).
# - urllib3 2.6.3: parents requests (>=1.26,<3) and others use loose floors; direct override
#   required for GHSA-mf9v-mfxr-j63j and GHSA-qccp-gfcp-xxvc until base image refreshes.
RUN conda run -n base python -m pip install --upgrade --no-cache-dir 'python-dotenv>=1.2.2' 'urllib3>=2.7.0'

# pip 26.0.1 in both base (python3.13) and ptca (python3.10) conda envs from ACPT base image needs upgrade
# (GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357; fixed in 26.1). pip is the package manager itself, bundled by conda;
# no parent package depends on it. We use `conda install` (not `pip install --upgrade pip`) so that conda-meta
# is updated alongside site-packages — vulnerability scanners read conda-meta and would otherwise still report
# the stale 26.0.1 version. Direct override is the only remediation until the ACPT base image ships pip>=26.1.
RUN conda install -n base -c conda-forge -y 'pip>=26.1.1' && \
    conda install -n ptca -c conda-forge -y 'pip>=26.1.1'

# py-rattler 0.23.2 (preinstalled in ACPT base via conda) ships a compiled rattler.abi3.so that
# bundles vulnerable Rust crates — openssl 0.10.75 (5 CRITICAL), aws-lc-sys 0.37.1
# (3 HIGH + 2 MEDIUM), bytes 1.11.0 (HIGH), tar 0.4.44 (HIGH + MEDIUM), rustls-webpki 0.103.9
# (4 MEDIUM), rand 0.8.5/0.9.2 (MEDIUM). py-rattler 0.23.2 is the latest release on PyPI and still
# ships the same vulnerable crates, so upgrading does not help. py-rattler is a conda packaging
# helper not used by Hugging Face NLP finetune workloads. This step runs AFTER all conda install /
# conda run steps so that conda's env-consistency checks cannot reinstall the .so from its package
# cache. We:
#   1. uninstall the package via conda + pip (to drop conda-meta and any pip RECORD);
#   2. delete the site-packages dir, any leftover py(_|-)rattler*.dist-info metadata, and any
#      stray rattler.abi3.so, scanning the full filesystem (no -xdev) so /opt/conda symlinked to
#      /opt/miniconda on a different mount is also covered;
#   3. delete the conda-meta JSON record so vulnerability scanners that read conda-meta no longer
#      list py-rattler as installed.
RUN (conda remove -n base -y py-rattler 2>/dev/null || true) && \
    (conda run -n base python -m pip uninstall -y py-rattler 2>/dev/null || true) && \
    find / \( -path '*/site-packages/rattler' -o -name 'rattler.abi3.so' -o -name 'py_rattler-*.dist-info' -o -name 'py-rattler-*.dist-info' \) -prune -exec rm -rf {} + 2>/dev/null; \
    find / -path '*/conda-meta/py-rattler-*.json' -delete 2>/dev/null; \
    find / -path '*/conda-meta/py_rattler-*.json' -delete 2>/dev/null; \
    true

# clean conda and pip caches
RUN rm -rf ~/.cache/pip
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/

