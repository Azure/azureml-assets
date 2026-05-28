#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root

# USN-8284-1/USN-8283-1: upgrade Ubuntu packages to patched Jammy versions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgnutls30 rsync \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10
# specforge 0.1.0 pins sglang==0.5.5; override needed after specforge install.
RUN pip install --no-cache-dir --force-reinstall "sglang>=0.5.10"

RUN pip install azureml-acft-common-components=={{latest-pypi-version}}
RUN pip install numpy==2.2.5
RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}

RUN pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0"
# wandb>=0.26.0: fixes Go stdlib vulnerabilities (GO-2026-4864/4865/4866/4869/4870/4946/4947)
# in bundled wandb-core binary (Go stdlib v1.26.1 -> v1.26.2)
RUN pip install --no-cache-dir --upgrade "wandb>=0.26.0"
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
RUN pip install xgrammar==0.1.32
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class;
# patched only in transformers>=5.0.0rc3. Upgrading to latest stable 5.x.
RUN pip install transformers==5.5.4
# Python 3.13 conda env overrides for vulnerable preinstalled transitive deps
# after checking latest parent metadata on 2026-05-25:
#  - python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94): metadata probes found
#    pydantic-settings 2.14.1 requires only >=0.21.0, uvicorn 0.48.0 requires
#    >=0.13 for [standard], and anaconda-auth 0.15.0 has no version floor.
#  - urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc / CVE-2026-44431, GHSA-mf9v-mfxr-j63j /
#    CVE-2026-44432): requests 2.34.2 still permits urllib3<3,>=1.26.
#  - idna>=3.15 (GHSA-65pc-fj4g-8rjx / CVE-2026-45409): metadata probes found
#    requests 2.34.2, anyio 4.13.0, httpx 0.28.1, and yarl 1.24.2 all still
#    permit idna below 3.15, so no parent upgrade forces the patched version.
#  - click>=8.3.3 (GHSA-47fr-3ffg-hgmw / CVE-2026-7246): metadata probes found
#    typer 0.25.1 requires only >=8.2.1 and anaconda-cli-base 0.8.2 has no floor.
#  - pip>=26.1 (GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357): shipped at 26.0.1 in both
#    Python envs. Pip self-upgrade leaves a stale
#    conda-meta/pip-26.0.1-*.json record that the SCA scanner still flags; remove
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
