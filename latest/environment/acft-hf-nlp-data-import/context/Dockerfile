# openmpi image
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260609.v2

USER root

# The base image's miniconda shipped Python 3.10 through tag 20260409.v4, but tag
# 20260507.v1 upgraded it to Python 3.12 (`MINICONDA_VERSION=py312_26.1.1-1` in the
# base image history). The packages installed below — azureml-acft-common-components
# and azureml-acft-contrib-hf-nlp — declare `Requires-Python >=3.8,<3.12` for every
# released version (latest 0.0.89, checked against PyPI 2026-05-11), so the build
# fails on the new base if we use the default base env directly. Until the upstream
# packages add Python 3.12 support we provision a dedicated Python 3.10 conda env
# at $AZUREML_CONDA_ENVIRONMENT_PATH and prepend it to PATH so subsequent `pip`
# calls target it. Same pattern is used by
# assets/training/automl/environments/ai-ml-automl-dnn-gpu/context/Dockerfile.
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-acft-hf-nlp-data-import
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# sudo is expected by Singularity inside the image
# Security: upgrade all OS packages and install security-patched system libraries.
# `apt-get -y upgrade` brings every base-image package to its latest noble-updates
# / noble-security version, so we only explicitly install packages that are NOT
# present in the openmpi5.0-ubuntu24.04 base image:
#   - sudo:        required by Singularity (see comment above)
#   - locales:     downstream Python locale support
#   - libssl-dev:  build-time headers for Python C extensions / wheel builds
#   - sqlite3:     CLI used by some downstream tooling
# Packages that USED to be in this list (libxml2, libc-bin, libc-dev, libc6,
# dpkg, dpkg-dev, libdpkg-perl, libssl3, openssl) were removed because they are
# all already installed by the base image and `apt-get -y upgrade` covers their
# security patches — re-listing them was redundant.
# openssh USN-8222-1 (>= 1:9.6p1-3ubuntu13.16): no longer needs an explicit
# `apt-get install --reinstall -y openssh-{client,server,sftp-server}` override.
# Verified 2026-05-12 that base mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04
# at the latest tag (20260507.v1) plus current noble-security state lets
# `apt-get -y upgrade` alone bring openssh to 1:9.6p1-3ubuntu13.16 (test build
# `:test-cleanup`, ACR run ca6n). openssh is shipped by the base image (no
# upstream parent package in this context), so the apt upgrade is the only fix.
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && \
    apt-get install -y sudo locales libssl-dev sqlite3 && \
    # Security: explicitly upgrade specific packages with known CVEs to ensure they reach
    # the required patched versions regardless of base-image state:
    #   libgnutls30t64 USN-8284-1: >=3.8.3-1.1ubuntu3.6
    #   libgcrypt20    USN-8319-1: >=1.10.3-2ubuntu0.1
    #   nginx/nginx-common/nginx-light USN-8354-1: >=1.24.0-2ubuntu7.9
    #   liblzma5/xz-utils USN-8362-1: >=5.6.1+really5.4.5-1ubuntu0.3
    apt-get install -y --only-upgrade \
        libgnutls30t64 \
        libgcrypt20 \
        nginx \
        nginx-common \
        nginx-light \
        liblzma5 \
        xz-utils \
    || true && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Security: upgrade pip in BASE miniconda (/opt/miniconda) to fix CVE-2026-6357
# (GHSA-jp4c-xjxw-mgf9). The base miniconda is independent of the Python 3.10 env
# created below; the vulnerability scanner reports pip from any Python installation
# in the image, so both must be patched. pip has no upstream parent package to bump,
# so a direct override is the only fix. Tag 20260507.v1+ already ships pip 26.1.1
# in the base, making this a no-op for newer bases.
# idna (GHSA-65pc-fj4g-8rjx): bump to >=3.15 in base miniconda (Python 3.12 env)
# since the scanner reports idna from opt/miniconda regardless of the conda env below.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'pip>=26.1' 'idna>=3.15' && rm -rf /root/.cache/pip

# Provision the Python 3.10 conda env. We don't constrain pip here because the
# `defaults` conda channel may not yet have pip 26.1+; we upgrade with `pip install
# --upgrade pip>=26.1` immediately after env creation (next RUN below).
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip -y && \
    conda clean -afy

COPY requirements.txt .

# Security: upgrade pip in the new Python 3.10 env to fix CVE-2026-6357
# (GHSA-jp4c-xjxw-mgf9). PATH is already prepended with the new env, so `pip`
# resolves to $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1'

RUN pip install -r requirements.txt --no-cache-dir

# setuptools==82.0.1, wheel==0.46.3, cryptography==46.0.5, urllib3==2.6.3, h2==4.3.0
# are already at fixed versions in the openmpi base image (20260315.v1).
# The override below only targets packages NOT fixed in base or pulled in vulnerable by requirements.txt.
# aiohttp (GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8): transitive dep of azure-core/datasets; bumped floor
#   to >=3.14.0 to resolve USN-reported CVEs (previous floor >=3.13.4 was insufficient).
# cryptography (GHSA-m959-cc7f-wv43, GHSA-p423-j2cm-9vmq): base image has 46.0.5; floor at 46.0.7+.
# requests (GHSA-gc5v-m9x4-r6x2): transitive dep of many packages; parents use loose floors.
# scikit-learn: explicit pin removed — azureml-acft-contrib-hf-nlp 0.0.89 already
#   pins `scikit-learn<1.6.0,>=1.5.1`, so the parent enforces the secure floor (>=1.5.1
#   ships the CVE-2024-5206 fix the historical pin protected against). Pip resolves to 1.5.2.
RUN pip install --no-cache-dir 'aiohttp>=3.14.0' 'requests>=2.33.0' 'cryptography>=46.0.7' && rm -rf /root/.cache/pip

# The below file is required for baking the code into the environment 
COPY data_import_run.py /azureml/data_import/run.py

# dummy number to change when needing to force rebuild without changing the definition: 4

