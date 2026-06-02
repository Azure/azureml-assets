FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.1

USER root

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 26.0.1 in both conda envs has GHSA-jp4c-xjxw-mgf9. pip is the bootstrap installer,
# not a runtime dependency from requirements.txt or their transitive deps. The current ACPT
# base tag (biweekly.202605.2) still ships pip 26.0.1 in conda-meta; conda-forge has 26.1.1
# and updates the conda metadata that the SBOM scanner reads. Use --no-deps to avoid pulling
# unrelated conda-forge package updates into the base image.
RUN /opt/conda/bin/conda install -n ptca -c conda-forge -y --no-deps 'pip>=26.1' && \
    /opt/conda/bin/conda install -n base -c conda-forge -y --no-deps 'pip>=26.1' && \
    /opt/conda/bin/conda clean -afy

# Install required packages from pypi
COPY requirements.txt .
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
RUN pip install --no-cache-dir --upgrade 'Mako>=1.3.11' 'GitPython>=3.1.47'

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
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'urllib3>=2.7.0' 'python-dotenv>=1.2.2' 'idna>=3.15' 'click>=8.3.3'
