# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Apply latest Ubuntu 22.04 security patches. Fresh rebuild pulls patched
# versions of curl/libcurl4/libcurl3-gnutls (USN-8227-1, 7.81.0-1ubuntu1.24),
# libnghttp2-14 (USN-8233-1, 1.43.0-1ubuntu0.3), kmod/libkmod2 (USN-8226-1,
# 29-1ubuntu1.1), sed (USN-8229-1, 4.8-1ubuntu2.1) and openssh-client
# (USN-8222-1, 1:8.9p1-3ubuntu0.15) directly from the Ubuntu security archive.
RUN apt-get -y update && apt-get -y upgrade

# pip 26.0/26.0.1 in both the base (py3.13) and ptca (py3.10) conda envs is
# vulnerable to GHSA-jp4c-xjxw-mgf9 (fixed in pip>=26.1). pip is a build/install
# tool installed by conda from the upstream base image
# (mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280, biweekly
# tag) — there is no parent Python package that brings it in, so an upstream
# parent upgrade is not possible. The base ACPT image has not yet refreshed to
# pip 26.1+ as of 2026-05-12, so we override here. We use `conda install`
# (rather than `pip install --upgrade`) so the conda-meta JSON and
# /opt/conda/pkgs cache are also updated, and we additionally remove stray
# pip-26.0*.dist-info / conda-meta entries from prior pip self-upgrades that
# conda does not track — otherwise the SBOM scanner re-flags them. Done before
# the requirements install so requirements are installed with the patched pip.
RUN conda install -y --solver=classic -n base -c conda-forge pip==26.1.1 && \
    conda install -y --solver=classic -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# pyOpenSSL 26.x remediates CVE-2026-27459/CVE-2026-27448; keep cryptography <47 for mlflow compatibility.
RUN pip install --no-cache-dir --upgrade 'pyopenssl>=26.0.0,<27' 'cryptography>=46,<47'

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# vulnerability overrides in base conda env (python3.13)
# setuptools: base image has 82.0.0, need >=82.0.1 for GHSA-58pv-8j8x-9vj2;
#   latest setuptools on PyPI as of 2026-05-12 is 82.0.1 (no parent package
#   pins setuptools to a CVE-safe floor), so direct upgrade is required.
# python-dotenv: transitive dep of pydantic-settings (requires python-dotenv>=0.21.0)
#   and anaconda-auth (requires python-dotenv with no version floor); even latest
#   pydantic-settings 2.14.1 and anaconda-auth 0.14.4 use the same loose floors,
#   so >=1.2.2 cannot be forced via parents (GHSA-mf9w-mj56-hr94).
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'python-dotenv>=1.2.2'
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
