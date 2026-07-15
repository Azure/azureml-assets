FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch/torchvision from the pytorch CUDA index (kept out of requirements.txt so the build
# context contains no disallowed extra pip index flag); fixes GHSA-qfhq-4f3w-5fph, GHSA-vgrw-7cvw-pwgx.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.10.0 torchvision==0.25.0
RUN pip install -r requirements.txt --no-cache-dir

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635, GHSA-fh64-r2vc-xvhr);
# upgrade after requirements install. azureml-mlflow 1.62.0.post2 pins mlflow-skinny<=3.9.0,
# so mlflow must be upgraded separately (post-pip-install) to avoid resolver conflict.
RUN pip install --no-cache-dir mlflow==3.11.1
# fastmcp + mcp were installed as regular deps of mlflow 3.5.0 (fastmcp→mcp) but orphaned after
# mlflow 3.11.1 (fastmcp moved to optional "mcp" extra); uninstall both to remove vulnerable packages
RUN pip uninstall -y fastmcp mcp

# Override vulnerable transitive deps in the ptca env (Python 3.10) that pip won't auto-upgrade:
# Mako: transitive dep (mlflow → alembic 1.18.4 → Mako); alembic 1.18.4 declares "Mako" with no
#   version pin at all, so pip resolves to 1.3.10 which has GHSA-v92g-xgxw-vvmm. No parent
#   release floors Mako >= 1.3.11, so explicit override is the only fix.
# GitPython: transitive dep (mlflow → mlflow-skinny 3.11.1 requires gitpython<4,>=3.1.9); the
#   loose floor lets pip resolve 3.1.46 which has GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4.
#   mlflow-skinny has no release that pins gitpython>=3.1.47, so explicit override is required.
# aiohttp: transitive dep; resolves to 3.13.5 which has GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg.
#   No parent release floors aiohttp>=3.14.0, so explicit override is required.
# pyarrow: resolves to 20.0.0 which has GHSA-rgxp-2hwp-jwgg; no parent release floors
#   pyarrow>=23.0.1, so explicit override is required.
# cryptography: transitive dep of ACPT base image and AzureML requirements; fixes GHSA-537c-gmf6-5ccf.
RUN pip install --no-cache-dir --upgrade 'Mako>=1.3.11' 'GitPython>=3.1.47' 'aiohttp>=3.14.0' 'pyarrow>=23.0.1' 'cryptography>=48.0.1'

# Upgrade vulnerable packages in the system Python (3.13)
# urllib3: transitive dep via requests; loose floor (urllib3<3,>=1.21.1) resolves to vulnerable 2.6.3
#   (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j). No parent release floors urllib3>=2.7.0, override required.
# python-dotenv: transitive dep via pydantic-settings (>=0.21.0 floor); parent uses loose floor so base resolves to
#   vulnerable 1.2.1 (GHSA-mf9w-mj56-hr94); override to >=1.2.2
# idna: transitive via requests (idna<4,>=2.5), anyio (idna>=2.8), httpx (no pin) — all loose floors
#   resolve to vulnerable 3.11 (GHSA-65pc-fj4g-8rjx / CVE-2026-45409). No parent release floors
#   idna>=3.15, so explicit override is required.
# click: transitive via mlflow-skinny (click<9,>=7.0), uvicorn (click>=7.0), typer (click>=8.2.1),
#   flask (click>=8.1.3) — all loose floors resolve to vulnerable 8.2.1 (GHSA-47fr-3ffg-hgmw /
#   CVE-2026-7246). No parent release floors click>=8.3.3, so explicit override is required.
# aiohttp: resolves to 3.13.5 which has GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg. No parent
#   release floors aiohttp>=3.14.0, so explicit override is required.
# PyJWT: resolves to 2.12.1 which has GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4. No parent
#   release floors PyJWT>=2.13.0, so explicit override is required.
# py-rattler: transitive dep of ACPT base image conda tooling; fixes GHSA-q53q-5r4j-5729.
# cryptography: transitive dep of ACPT base image and AzureML requirements; fixes GHSA-537c-gmf6-5ccf.
# pydantic-settings: transitive dep of AzureML requirements that can downgrade the patched base; fixes GHSA-4xgf-cpjx-pc3j.
# msgpack: transitive dep of AzureML/MLflow requirements that can downgrade the patched base; fixes GHSA-6v7p-g79w-8964.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'urllib3>=2.7.0' 'python-dotenv>=1.2.2' 'idna>=3.15' 'click>=8.3.3' 'aiohttp>=3.14.0' 'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=48.0.1' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1'
