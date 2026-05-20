# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

# Pull latest Ubuntu 22.04 security patches on each rebuild.
RUN apt-get -y update && apt-get -y upgrade

# pip in both conda envs (base py3.13 and ptca py3.10) ships as 26.0.1 in the
# upstream ACPT base image, which is vulnerable to GHSA-jp4c-xjxw-mgf9 (fixed
# in pip>=26.1). pip is a build/install tool with no Python parent package —
# conda is its installer — so an upstream parent upgrade is not possible. The
# ACPT base has not yet refreshed to pip 26.1+ as of 2026-05-19, so we override
# here. We use `conda install` so the conda-meta JSON and /opt/conda/pkgs cache
# are also updated, and remove stray pip-26.0* dist-info / conda-meta entries
# from prior pip self-upgrades that conda does not track — otherwise the SBOM
# scanner re-flags them. Done before the requirements install so requirements
# are installed with the patched pip.
RUN conda install -y --solver=classic -n base -c conda-forge pip==26.1.1 && \
    conda install -y --solver=classic -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Install required packages (ptca env). setuptools>=82.0.1 patches
# GHSA-58pv-8j8x-9vj2 in the ptca env which ships setuptools 81.0.0.
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# Vulnerability overrides in base conda env (python3.13):
# - setuptools: base ships 82.0.0; need >=82.0.1 for GHSA-58pv-8j8x-9vj2.
#   Latest setuptools on PyPI (82.0.1) and no upstream package pins setuptools
#   to a CVE-safe floor, so a direct upgrade is the only available fix.
# - python-dotenv: base ships 1.2.1; need >=1.2.2 for GHSA-mf9w-mj56-hr94. It
#   is a transitive dep of pydantic-settings 2.12.0 (requires
#   python-dotenv>=0.21.0) and anaconda-auth 0.14.4 (no version floor); even
#   the latest releases of both still carry loose floors, so >=1.2.2 cannot
#   be forced via parents.
# - urllib3: base ships 2.6.3; need >=2.7.0 for GHSA-qccp-gfcp-xxvc and
#   GHSA-mf9v-mfxr-j63j. The only parent in the base env is requests 2.33.1
#   with constraint `urllib3<3,>=1.26`; the latest requests release does not
#   raise this floor either, so a parent upgrade cannot pull in urllib3>=2.7.0
#   and a direct override is required.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
        'setuptools>=82.0.1' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0'

# Vulnerability overrides in ptca conda env (python3.10):
# - urllib3: the ptca env ships urllib3 2.7.0 from the base image, but the
#   `conda install pip==26.1.1` step above re-solves urllib3 back down to
#   2.6.3 (vulnerable to GHSA-qccp-gfcp-xxvc / GHSA-mf9v-mfxr-j63j). Parents
#   (requests 2.34.2, torchdata, fsspec) all use loose floors (>=1.25 / >=1.26)
#   so they cannot force urllib3>=2.7.0; direct override is required.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
        'urllib3>=2.7.0'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
