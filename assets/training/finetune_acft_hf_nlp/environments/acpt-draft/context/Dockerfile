#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root

# Apply latest Ubuntu 22.04 (jammy) security patches. The ACPT base image
# (mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280, biweekly
# tag) ships unpatched OS packages flagged by the SBOM scanner:
#   - sed 4.8-1ubuntu2 -> 4.8-1ubuntu2.1 (USN-8229-1)
#   - curl/libcurl4/libcurl3-gnutls 7.81.0-1ubuntu1.23 -> 7.81.0-1ubuntu1.24 (USN-8227-1)
#   - libnghttp2-14 1.43.0-1ubuntu0.2 -> 1.43.0-1ubuntu0.3 (USN-8233-1)
# These come straight from the base image (no parent package in this context)
# and the only fix is `apt-get upgrade` to pull jammy-security versions. Doing
# a full `apt-get -y upgrade` (rather than pinning each package) also picks up
# any other Ubuntu security updates in one step.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 26.0.1 in both the base (py3.13) and ptca (py3.10) conda envs is
# vulnerable to GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357 (fixed in pip>=26.1).
# pip is a build/install tool installed by conda from the upstream base image
# (mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280, biweekly
# tag) — there is no parent Python package that brings it in, so an upstream
# parent upgrade is not possible. The ACPT base image has not yet refreshed to
# pip 26.1+ as of 2026-05-12 (latest pip on PyPI is 26.1.1, available on
# conda-forge as pip-26.1.1-pyh*_0 since 2026-05-05). We use `conda install`
# (rather than `pip install --upgrade`) so the conda-meta JSON is also
# updated, and we additionally remove stray pip-26.0*.dist-info / conda-meta
# entries from prior pip self-upgrades that conda does not track — otherwise
# the SBOM scanner re-flags them. Done before the requirements install so
# subsequent `pip install` calls run with the patched pip.
RUN conda install -y --solver=classic -n base -c conda-forge pip==26.1.1 && \
    conda install -y --solver=classic -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10
# specforge 0.1.0 pins sglang==0.5.5; override needed after specforge install.
RUN pip install --no-cache-dir --force-reinstall "sglang>=0.5.10"

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}

# following are for vulnerability overrides at later
# release of following packages consider moving then to requirements.txt
RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
# wandb>=0.26.0: fixes Go stdlib vulnerabilities (GO-2026-4864/4865/4866/4869/4870/4946/4947)
# in bundled wandb-core binary (Go stdlib v1.26.1 -> v1.26.2)
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0"
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
RUN pip install xgrammar==0.1.32
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class;
# patched only in transformers>=5.0.0rc3. Upgrading to latest stable 5.x.
RUN pip install transformers==5.5.4
# python-dotenv>=1.2.2: GHSA-mf9w-mj56-hr94; transitive dep in the BASE conda env (python3.13)
#   shipped by the ACPT base image at 1.2.1. Parent packages are pydantic-settings
#   (2.14.0 still requires only python-dotenv>=0.21.0), uvicorn (0.46.0 standard extra
#   requires python-dotenv>=0.13), and fastmcp (3.2.4 requires python-dotenv>=1.1.0) —
#   all use loose floors and no released parent version forces >=1.2.2, so direct override
#   in the base conda env is the only fix path.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
