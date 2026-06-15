# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.2

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities.
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 26.0.1 in both the ptca conda env (Python 3.10) and the /opt/conda base env (Python 3.13)
# has GHSA-jp4c-xjxw-mgf9 (fixed in pip>=26.1). pip is the bootstrap installer — no Python
# package in requirements.txt or any of their transitive deps declares pip as a runtime dep
# (verified against requires_dist of azureml-acft-*, mlflow, mlflow-skinny, transformers,
# torch, etc.), so an upstream parent-package upgrade is not possible. The conda `defaults`
# channel only has up to pip 26.0.1 (conda-forge has 26.1.1 but is not enabled in the base
# image), so `conda install pip==26.1.1` fails — we use `pip install --upgrade` (from PyPI)
# and then explicitly remove the stale conda-meta JSON for pip 26.0.1 so the SBOM scanner
# does not re-flag it. Done before pip-installing requirements so all subsequent installs
# use the patched pip.
RUN pip install --no-cache-dir --upgrade 'pip==26.1.1' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'pip==26.1.1' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Override vulnerable transitive deps in the ptca env (Python 3.10) that pip won't auto-upgrade:
# Mako: transitive dep (mlflow → alembic 1.18.4 → Mako); alembic 1.18.4 declares "Mako" with no
#   version pin at all, so pip resolves to 1.3.10 which has GHSA-v92g-xgxw-vvmm. No parent
#   release floors Mako >= 1.3.11, so explicit override is the only fix.
# GitPython: transitive dep (mlflow → mlflow-skinny 3.11.1 requires gitpython<4,>=3.1.9); the
#   loose floor lets pip resolve 3.1.46 which has GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4.
#   mlflow-skinny has no release that pins gitpython>=3.1.47, so explicit override is required.
# pytest: shipped by the ACPT base image (NOT pulled in by any package in requirements.txt nor
#   any of their transitive deps; verified by inspecting requires_dist of mlflow, mlflow-skinny,
#   transformers, azureml-acft-*, etc.). No parent package to bump — the base image pre-installs
#   pytest 7.4.3 and the ACPT base hasn't been rebuilt with a fix yet, so explicit override
#   to >=9.0.3 is required (GHSA-6w46-j5rx-g56g).
# skops: transitive dep (mlflow → mlflow-skinny 3.11.1 requires skops<1 with no minimum), pip
#   resolves to 0.11.0 which has CVE-2025-54412/54413/54886 (arbitrary code execution).
#   mlflow-skinny has no release that bumps the floor, so explicit override is required.
# urllib3: transitive dep (requests>=2.33.0 → urllib3<3,>=1.26); latest requests 2.34.2 still
#   declares `urllib3<3,>=1.26`, no parent package release floors urllib3>=2.7.0, so explicit
#   override is the only fix for GHSA-qccp-gfcp-xxvc (CVE-2026-44431) and GHSA-mf9v-mfxr-j63j
#   (CVE-2026-44432).
RUN pip install --no-cache-dir --upgrade 'Mako>=1.3.11' 'GitPython>=3.1.47' 'pytest>=9.0.3' 'skops>=0.13.0' 'urllib3>=2.7.0'

# python-dotenv: pre-installed in the /opt/conda base env (Python 3.13) by the ACPT base image
#   at version 1.2.1 which has GHSA-mf9w-mj56-hr94. Not pulled in by any package in
#   requirements.txt nor their transitive deps, so the ptca env (Python 3.10) gets the
#   already-patched 1.2.2 from the base image's own upgrades. The /opt/conda Python 3.13 env
#   is unaffected by `pip install -r requirements.txt` (which targets ptca's pip), so an
#   explicit upgrade against /opt/conda/bin/python3.13 is required to reach >= 1.2.2.
# urllib3: same situation — /opt/conda Python 3.13 ships urllib3 2.6.3 from the base image and
#   is not reached by ptca pip installs, so an explicit upgrade to >=2.7.0 is required to fix
#   GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j in this env.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'urllib3>=2.7.0'
