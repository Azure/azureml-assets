FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}
USER root
ENV PIP_PROGRESS_BAR=off
ENV PIP_CONSTRAINT=/tmp/security-constraints.txt

# Apply OS security updates inherited from the base image.
# libpam packages are inherited from the base image; fixes USN-8601-1.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpam-runtime=1.4.0-11ubuntu2.7 \
        libpam0g=1.4.0-11ubuntu2.7 \
        libpam-modules=1.4.0-11ubuntu2.7 \
        libpam-modules-bin=1.4.0-11ubuntu2.7 \
        libgnutls30 rsync \
        dotnet-host-8.0 dotnet-hostfxr-8.0 dotnet-runtime-8.0 \
        libgcrypt20 \
        liblzma5 xz-utils \
        wget=1.21.2-2ubuntu1.4 \
        tar=1.34+dfsg-1ubuntu0.1.22.04.6 \
        libkrb5support0=1.19.2-2ubuntu0.8 \
        libgssapi-krb5-2=1.19.2-2ubuntu0.8 \
        libk5crypto3=1.19.2-2ubuntu0.8 \
        libkrb5-3=1.19.2-2ubuntu0.8 \
        libsqlite3-0=3.37.2-2ubuntu0.7 \
        libasound2=1.2.6.1-1ubuntu1.2 \
        libasound2-data=1.2.6.1-1ubuntu1.2 \
        openssh-client=1:8.9p1-3ubuntu0.16 \
 && rm -rf /var/lib/apt/lists/*

COPY security-constraints.txt /tmp/security-constraints.txt
COPY requirements.txt .
# GHSA-jx93-g359-86wm, GHSA-hvwj-8w5g-28rg: sglang vulnerabilities; patched in >=0.5.10
# specforge 0.1.0 pins sglang==0.5.5; override needed after specforge install.
# Upgrade wandb and remove its bundled wandb-core Go binary; the binary carries
# vulnerable Go modules (stdlib/grpc/x/text) and wandb falls back to its Python backend.
# torch is a transitive dep of SpecForge/sglang; this image's pip layers override the
# base torch 2.8.0+cu126 and must stay above GHSA-rrmf-rvhw-rf47's patched floor.
# GHSA-7rgv-gqhr-fxg3: xgrammar stack exhaustion DoS; patched in 0.1.32
# GHSA-69w3-r845-3855 (CVE-2026-1839): arbitrary code execution in Trainer class;
# patched only in transformers>=5.0.0rc3. Upgrading to latest stable 5.x.
# GHSA-86qp-5c8j-p5mr: starlette vulnerability; patched in >=1.0.1 (ptca env).
# GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8: aiohttp vulnerabilities; patched in >=3.14.0.
# Upgrade in both the base (Python 3.14) and ptca (Python 3.10) conda envs.
# Python 3.14 conda env overrides for vulnerable preinstalled transitive deps:
#  - python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94): metadata probes found
#    pydantic-settings 2.14.2 requires only >=0.21.0, uvicorn 0.48.0 requires
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
#  - PyJWT>=2.13.0 (GHSA-993g-76c3-p5m4, GHSA-jq35-7prp-9v3f): two CVEs fixed in 2.13.0;
#    installed as transitive dep at 2.12.1 in both Python envs.
#  - py-rattler>=0.24.0 (GHSA-q53q-5r4j-5729): pinned transitive dep brought in by
#    base image in the Python 3.13 conda env.
#  - cryptography>=48.0.1 (GHSA-537c-gmf6-5ccf): pinned transitive dep brought in by
#    the base image in the Python 3.13 conda env.
#  - pydantic-settings>=2.14.2 (GHSA-4xgf-cpjx-pc3j): pinned transitive dep of
#    pydantic/Anaconda auth tooling; this image's pip layers can override the base env.
#  - msgpack>=1.2.1 (GHSA-6v7p-g79w-8964): pinned transitive dep of MLflow/AzureML
#    MLflow packages; remove stale conda records after pip upgrade.
#  - setuptools>=83.0.0 (GHSA-h35f-9h28-mq5c): pinned transitive dep brought in
#    by the base conda envs; remove stale conda records after pip upgrade.
# Remove pip's vendored SBOM files after pip upgrades; they describe pip's vendored
# sources, not installed runtime packages, and cause stale SCA package records.
RUN pip install -r requirements.txt --no-cache-dir \
 && pip install --no-cache-dir --force-reinstall "sglang>=0.5.10" \
 && pip install --no-cache-dir azureml-acft-common-components=={{latest-pypi-version}} \
 && pip install --no-cache-dir numpy==2.2.5 \
 && pip install --no-cache-dir azureml-evaluate-mlflow=={{latest-pypi-version}} \
 && pip install --no-cache-dir --force-reinstall "mlflow>=3.2.0,<4.0.0" \
 && pip install --no-cache-dir --upgrade "wandb>=0.28.1" \
 && find /opt/conda/envs/ptca -name 'wandb-core' -path '*/wandb/bin/*' -delete \
 && pip install --no-cache-dir --upgrade "torch>=2.13.0" \
 && pip install --no-cache-dir xgrammar==0.1.32 \
 && pip install --no-cache-dir transformers==5.5.4 \
 && pip install --no-cache-dir 'starlette>=1.0.1' \
 && pip install --no-cache-dir 'aiohttp>=3.14.0' \
 && conda run -n base python -m pip install --no-cache-dir 'aiohttp>=3.14.0' \
 && conda run -n base python -m pip install --no-cache-dir --upgrade \
        'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'pip>=26.1' \
        'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=48.0.1' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1' \
        'setuptools>=83.0.0' \
 && pip install --no-cache-dir --upgrade 'pip>=26.1' 'PyJWT>=2.13.0' 'msgpack>=1.2.1' 'setuptools>=83.0.0' \
 && find /opt/conda /opt/conda/envs/ptca -path '*/conda-meta/pip-*.json' -delete \
 && find /opt/conda /opt/conda/envs/ptca \( -path '*/conda-meta/setuptools-*.json' -o -path '*/conda-meta/msgpack-python-*.json' \) -delete \
 && find /opt/conda /opt/conda/envs/ptca -path '*/site-packages/pip/_vendor/bom.cdx.json' -delete \
 && rm -rf ~/.cache/pip /root/.cache/pip /tmp/pip-*
COPY loss /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/loss.py
COPY eagle3 /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/core/eagle3.py
COPY parse /opt/conda/envs/ptca/lib/python3.10/site-packages/specforge/data/parse.py
