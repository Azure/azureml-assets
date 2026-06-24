# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.2.v1

# Pull latest Ubuntu 22.04 security patches on each rebuild.
# Explicit package list ensures USN-8298-1 (.NET), USN-8319-1 (libgcrypt20),
# and USN-8362-1 (xz-utils/liblzma5) patches are applied even when apt-get
# upgrade would otherwise hold them back.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --only-upgrade \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        libgcrypt20 \
        liblzma5 \
        xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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
# - aiohttp: base ships 3.13.5; need >=3.14.0 for GHSA-jg22-mg44-37j8 and
#   GHSA-hg6j-4rv6-33pg. No parent in the base env pins aiohttp to a safe
#   floor, so a direct upgrade is required.
# - click: base ships 8.2.1; need >=8.3.3 for GHSA-47fr-3ffg-hgmw. No parent
#   in the base env forces click>=8.3.3, so a direct override is required.
# - idna: base ships 3.11; need >=3.15 for GHSA-65pc-fj4g-8rjx. Transitive
#   dep of httpx, requests, and others which all use loose floors; direct
#   override is required.
# - PyJWT: base ships 2.12.1; need >=2.13.0 for GHSA-993g-76c3-p5m4 and
#   GHSA-jq35-7prp-9v3f. No parent in the base env pins PyJWT to a safe
#   floor, so a direct upgrade is required.
# - py-rattler: pinned transitive dep of base conda/rattler tooling brought in
#   by the base image; fixes GHSA-q53q-5r4j-5729.
# - cryptography: pinned transitive dep brought in by the ACPT base image;
#   fixes GHSA-537c-gmf6-5ccf.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
        'setuptools>=82.0.1' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' \
        'aiohttp>=3.14.0' 'click>=8.3.3' 'idna>=3.15' 'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' 'cryptography>=48.0.1'

# Vulnerability overrides in ptca conda env (python3.10):
# - urllib3: the ptca env ships urllib3 2.7.0 from the base image, but the
#   `conda install pip==26.1.1` step above re-solves urllib3 back down to
#   2.6.3 (vulnerable to GHSA-qccp-gfcp-xxvc / GHSA-mf9v-mfxr-j63j). Parents
#   (requests 2.34.2, torchdata, fsspec) all use loose floors (>=1.25 / >=1.26)
#   so they cannot force urllib3>=2.7.0; direct override is required.
# - aiohttp: ptca env ships 3.13.5; need >=3.14.0 for GHSA-jg22-mg44-37j8 and
#   GHSA-hg6j-4rv6-33pg. No parent pins aiohttp to a safe floor; direct
#   override is required.
# - starlette: ptca env ships 0.52.1; need >=1.0.1 for GHSA-86qp-5c8j-p5mr.
#   No parent in the ptca env forces starlette>=1.0.1; direct override is
#   required.
# - click: ptca env ships 8.2.1; need >=8.3.3 for GHSA-47fr-3ffg-hgmw. No
#   parent in the ptca env forces click>=8.3.3; direct override is required.
# - pyarrow: ptca env ships 20.0.0; need >=23.0.1 for GHSA-rgxp-2hwp-jwgg.
#   No parent in the ptca env forces pyarrow>=23.0.1; direct override is
#   required.
# - cryptography: pinned transitive dep brought in by the ACPT base image;
#   fixes GHSA-537c-gmf6-5ccf.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
        'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'starlette>=1.0.1' 'click>=8.3.3' \
        'pyarrow>=23.0.1' 'cryptography>=48.0.1'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
