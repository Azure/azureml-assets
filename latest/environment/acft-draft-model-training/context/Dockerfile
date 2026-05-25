#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2
USER root

# OS package security fixes not yet present in the base image
# (USN-8251-1 libpng, USN-8229-1 sed, USN-8227-1 curl, USN-8233-1 nghttp2, USN-8284-1 gnutls, USN-8283-1 rsync)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpng16-16 sed curl libcurl4 libcurl3-gnutls libnghttp2-14 libgnutls30 rsync \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10
# specforge 0.1.0 pins sglang==0.5.5; override needed after specforge install.
RUN pip install --no-cache-dir --force-reinstall "sglang>=0.5.10"

RUN pip install azureml-acft-common-components==0.0.91
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow==0.0.91

RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
# wandb>=0.26.0: fixes Go stdlib vulnerabilities (GO-2026-4864/4865/4866/4869/4870/4946/4947)
# in bundled wandb-core binary (Go stdlib v1.26.1 -> v1.26.2)
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0"
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
RUN pip install xgrammar==0.1.32
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class;
# patched only in transformers>=5.0.0rc3. Upgrading to latest stable 5.x.
RUN pip install transformers==5.5.4
# Base conda env (python3.13) overrides for vulnerable transitive deps:
#  - python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94): pydantic-settings 2.14.1 still requires
#    only python-dotenv>=0.21.0; uvicorn[standard] requires >=0.13; fastmcp requires
#    >=1.1.0. No released parent forces >=1.2.2, so direct override is the only fix.
#  - urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc / CVE-2026-44431, GHSA-mf9v-mfxr-j63j /
#    CVE-2026-44432): transitive dep at 2.6.3 in base env.
#  - idna>=3.15 (GHSA-65pc-fj4g-8rjx / CVE-2026-45409): transitive dep at 3.11 in base
#    env; pulled by requests/urllib3/cryptography which all use loose floors (idna<4)
#    so no parent upgrade forces >=3.15.
#  - click>=8.3.3 (GHSA-47fr-3ffg-hgmw / CVE-2026-7246): transitive dep at 8.2.1 in
#    base env; pulled by mlflow/uvicorn/typer with loose floors (click>=8.x), so no
#    parent upgrade forces >=8.3.3.
#  - pip>=26.1 (GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357): shipped at 26.0.1 in both base
#    (python3.13) and PTCA (python3.10) envs. Pip self-upgrade leaves a stale
#    conda-meta/pip-26.0.1-*.json record that the SCA scanner still flags — remove
#    it explicitly after the upgrade.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
        'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'pip>=26.1' \
 && pip install --no-cache-dir --upgrade 'pip>=26.1' \
 && rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0.1-*.json \
          /opt/conda/conda-meta/pip-26.0.1-*.json
# clean conda and pip caches
RUN rm -rf ~/.cache/pip
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
