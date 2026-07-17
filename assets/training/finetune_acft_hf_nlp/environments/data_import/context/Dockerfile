FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

USER root

# Keep this environment on Python 3.10 to preserve the existing component runtime
# behavior while the base image's miniconda uses Python 3.12.
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-acft-hf-nlp-data-import
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# sudo is expected by Singularity inside the image
# Security: upgrade all OS packages and install security-patched system libraries.
# `apt-get -y upgrade` brings every base-image package to its latest noble-updates
# / noble-security version. The explicit installs below add packages that are NOT
# present in the openmpi5.0-ubuntu24.04 base image:
#   - sudo:        required by Singularity (see comment above)
#   - locales:     downstream Python locale support
#   - libssl-dev:  build-time headers for Python C extensions / wheel builds
#   - sqlite3:     CLI used by some downstream tooling
# Packages that USED to be in this list (libxml2, libc-bin, libc-dev, libc6,
# dpkg, dpkg-dev, libdpkg-perl, libssl3, openssl) were removed because they are
# all already installed by the base image and `apt-get -y upgrade` covers their
# security patches — re-listing them was redundant.
# The final `--only-upgrade` list documents security-sensitive OS packages that
# must remain at the current noble-security patch level even when inherited from
# the base image.
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && \
apt-get install -y sudo locales libssl-dev sqlite3 && \
apt-get install -y --only-upgrade \
    curl \
    libcurl4t64 \
    libcurl3t64-gnutls \
    gzip \
    libgnutls30t64 \
    libgcrypt20 \
    nginx \
        nginx-common \
        nginx-light \
        liblzma5 \
        xz-utils \
    && \
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

# setuptools==82.0.1, wheel==0.46.3, urllib3==2.6.3, h2==4.3.0 are already
# at fixed versions in the openmpi base image. The override below only targets
# packages NOT fixed in base or pulled in vulnerable by requirements.txt.
# aiohttp (GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8): transitive dep of azure-core/datasets; bumped floor
#   to >=3.14.0 to resolve USN-reported CVEs (previous floor >=3.13.4 was insufficient).
# requests (GHSA-gc5v-m9x4-r6x2): transitive dep of many packages; parents use loose floors.
# scikit-learn: explicit pin removed — azureml-acft-contrib-hf-nlp 0.0.89 already
#   pins `scikit-learn<1.6.0,>=1.5.1`, so the parent enforces the secure floor (>=1.5.1
#   ships the CVE-2024-5206 fix the historical pin protected against). Pip resolves to 1.5.2.
# pyarrow (GHSA-rgxp-2hwp-jwgg): transitive dep; bump to >=23.0.1 to fix the CVE.
# cryptography: pinned transitive dep pulled by azureml-acft-contrib-hf-nlp/base image; fixes GHSA-537c-gmf6-5ccf.
RUN pip install --no-cache-dir 'aiohttp>=3.14.0' 'requests>=2.33.0' 'cryptography>=48.0.1' 'pyarrow>=23.0.1' && rm -rf /root/.cache/pip

# The below file is required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
