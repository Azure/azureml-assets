FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities.
RUN apt-get -y update && apt-get -y upgrade && apt-get -y install unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 26.0.1 -> 26.1.1 in both envs to fix GHSA-jp4c-xjxw-mgf9 (bootstrap installer, no parent
# to bump). Installed from PyPI; stale conda-meta JSON removed so the SBOM scanner doesn't
# re-flag it. Done before installing requirements so subsequent installs use the patched pip.
RUN pip install --no-cache-dir --upgrade 'pip==26.1.1' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'pip==26.1.1' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# ptca-env (py3.10) transitive deps pip won't auto-upgrade; no parent floors the safe versions:
# Mako>=1.3.11 (GHSA-v92g-xgxw-vvmm), GitPython>=3.1.47 (GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4),
# pytest>=9.0.3 (GHSA-6w46-j5rx-g56g), skops>=0.13.0 (CVE-2025-54412/54413/54886),
# urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j), aiohttp>=3.14.0 (GHSA-hg6j-4rv6-33pg,
# GHSA-jg22-mg44-37j8), pyarrow>=23.0.1 (GHSA-rgxp-2hwp-jwgg).
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-537c-gmf6-5ccf.
RUN pip install --no-cache-dir --upgrade 'Mako>=1.3.11' 'GitPython>=3.1.47' 'pytest>=9.0.3' 'skops>=0.13.0' 'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'pyarrow>=23.0.1' 'cryptography>=48.0.1'

# base-env (py3.13) transitive deps not reached by the ptca pip installs; explicit upgrades required:
# python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94), urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j),
# idna>=3.15 (GHSA-65pc-fj4g-8rjx), click>=8.3.3 (GHSA-47fr-3ffg-hgmw), aiohttp>=3.14.0
# (GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8), PyJWT>=2.13.0 (GHSA-jq35-7prp-9v3f, GHSA-993g-76c3-p5m4).
# py-rattler: pinned transitive dep of ACPT base image conda tooling; brought in by base image, fixes GHSA-q53q-5r4j-5729.
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-537c-gmf6-5ccf.
# pydantic-settings: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-4xgf-cpjx-pc3j.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'aiohttp>=3.14.0' 'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=48.0.1' 'pydantic-settings>=2.14.2'
