# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

USER root

# Install unzip and upgrade OS packages to pick up any pending security fixes from the base image.
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

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
# pip: shipped by the base image conda env at 26.0.1 (GHSA-jp4c-xjxw-mgf9). pip is a top-level
#   tool with no parent package; the base image hasn't been rebuilt with pip 26.1 yet, so we
#   upgrade it explicitly in both the ptca env and the system Python 3.13 env below.
# urllib3: transitive dep via requests/botocore/etc. (requests pins urllib3<3,>=1.21.1 with a
#   loose floor); pip resolves to vulnerable 2.6.3 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j).
#   No parent release floors urllib3>=2.7.0, so explicit override is required.
# Mako: transitive dep (mlflow → alembic 1.18.4 → Mako); alembic 1.18.4 declares "Mako" with no
#   version pin at all, so pip resolves to 1.3.10 which has GHSA-v92g-xgxw-vvmm. No parent
#   release floors Mako >= 1.3.11, so explicit override is the only fix.
# GitPython: transitive dep (mlflow → mlflow-skinny 3.11.1 requires gitpython<4,>=3.1.9); the
#   loose floor lets pip resolve 3.1.46 which has GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4.
#   mlflow-skinny has no release that pins gitpython>=3.1.47, so explicit override is required.
# pytest: shipped by the ACPT base image (NOT pulled in by any package in requirements.txt nor
#   any of their transitive deps; verified by inspecting requires_dist of mlflow, mlflow-skinny,
#   azureml-mlflow, transformers, etc.). No parent package to bump — the base image pre-installs
#   pytest 7.4.3 and the ACPT base hasn't been rebuilt with a fix yet, so explicit override
#   to >=9.0.3 is required (GHSA-6w46-j5rx-g56g).
RUN pip install --no-cache-dir --upgrade 'pip>=26.1' 'urllib3>=2.7.0' 'Mako>=1.3.11' 'GitPython>=3.1.47' 'pytest>=9.0.3'

# Upgrade vulnerable packages in the system Python (3.13)
# pip: base image ships 26.0.1 (GHSA-jp4c-xjxw-mgf9); pip is a top-level tool with no parent, override required.
# urllib3: transitive dep via requests; loose floor (urllib3<3,>=1.21.1) resolves to vulnerable 2.6.3
#   (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j). No parent release floors urllib3>=2.7.0, override required.
# python-dotenv: transitive dep via pydantic-settings (>=0.21.0 floor); parent uses loose floor so base resolves to
#   vulnerable 1.2.1 (GHSA-mf9w-mj56-hr94); override to >=1.2.2
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'pip>=26.1' 'urllib3>=2.7.0' 'python-dotenv>=1.2.2'
