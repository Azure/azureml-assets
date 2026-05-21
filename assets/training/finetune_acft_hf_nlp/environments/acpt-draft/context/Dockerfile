#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root

# USN-8227-1 (curl/libcurl4/libcurl3-gnutls), USN-8233-1 (libnghttp2-14),
# USN-8251-1 (libpng16-16), USN-8229-1 (sed): Ubuntu 22.04 system packages
# installed in the base image. The base image tag already pins to the latest
# biweekly build, but it has not yet picked up these jammy-security updates.
# Direct apt upgrade is the only fix path.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# GHSA-jp4c-xjxw-mgf9 (CVE-2026-6357): pip<26.1 self-update vulnerability.
# pip has no parent package — it is shipped directly by the base image in both
# conda envs (base: python3.13 / ptca: python3.10), so we upgrade both via
# conda-forge and remove the leftover dist-info / conda-meta entries from the
# old 26.0.1 so the SBOM scanner does not double-detect the vulnerable version.
RUN conda install -y -n base -c conda-forge pip==26.1.1 && \
    conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# GHSA-qccp-gfcp-xxvc (CVE-2026-44431), GHSA-mf9v-mfxr-j63j (CVE-2026-44432):
# urllib3 streaming-API vulnerabilities; patched in urllib3>=2.7.0. Only the
# base conda env (python3.13) ships the vulnerable urllib3 2.6.3 as a transitive
# dep (pulled by requests/other HTTP clients in the base image). No parent
# package pins urllib3<2.7.0 with a tight upper bound, so a direct upgrade in
# the base env is the simplest fix.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade 'urllib3>=2.7.0'

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
