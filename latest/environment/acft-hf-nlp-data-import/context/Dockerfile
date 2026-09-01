FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260810.v1

USER root

# Keep this environment on Python 3.10 to preserve the existing component runtime
# behavior while the base image's miniconda uses Python 3.12.
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-acft-hf-nlp-data-import
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# sudo is expected by Singularity inside the image.
# Upgrade OS packages to the latest noble-updates/noble-security versions, then
# keep security-sensitive inherited packages at the current patched level.
RUN apt-get update && ACCEPT_EULA=Y apt-get -y upgrade && \
apt-get install -y sudo locales libssl-dev sqlite3 && \
apt-get install -y --only-upgrade \
    curl \
    gzip \
    libcurl3t64-gnutls \
    libcurl4t64 \
    libgcrypt20 \
    libgnutls30t64 \
    libgssapi-krb5-2 \
    libk5crypto3 \
    libkrb5-3 \
    libkrb5support0 \
    libssl-dev \
    libssl3t64 \
    libsqlite3-0 \
    nginx \
    nginx-common \
    nginx-light \
    openssl \
    sqlite3 \
    tar \
    wget \
    liblzma5 \
    xz-utils \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Remove pip's informational vendored-version manifests because VCM can report
# stale vendored package entries even when the installed packages are patched.
RUN find /opt/miniconda \( -path '*/pip/_vendor/vendor.txt' -o -path '*/pip/_vendor/bom.cdx.json' \) -delete && \
    rm -rf /root/.cache/pip

# Provision the Python 3.10 conda env. The conda channel now provides a patched
# pip, so no post-create pip upgrade is needed.
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip -y && \
    find $AZUREML_CONDA_ENVIRONMENT_PATH \( -path '*/pip/_vendor/vendor.txt' -o -path '*/pip/_vendor/bom.cdx.json' \) -delete && \
    conda clean -afy

COPY requirements.txt .

# wheel==0.46.3, urllib3==2.6.3, h2==4.3.0 are already
# at fixed versions in the openmpi base image. The override below only targets
# packages NOT fixed in base or pulled in vulnerable by requirements.txt.
# aiohttp (GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8): transitive dep of azure-core/datasets; bumped floor
#   to >=3.14.0 to resolve USN-reported CVEs (previous floor >=3.13.4 was insufficient).
# requests (GHSA-gc5v-m9x4-r6x2): transitive dep of many packages; parents use loose floors.
# setuptools (GHSA-5rjg-fvgr-3xxf, GHSA-h35f-9h28-mq5c): transitive dep of azureml_acft_common_components/
#   azureml-acft-contrib-hf-nlp; requirements install can downgrade the Python 3.10 env.
# msgpack (GHSA-6v7p-g79w-8964): transitive dep of azureml_acft_common_components/azureml-acft-contrib-hf-nlp.
# scikit-learn: explicit pin removed — azureml-acft-contrib-hf-nlp 0.0.89 already
#   pins `scikit-learn<1.6.0,>=1.5.1`, so the parent enforces the secure floor (>=1.5.1
#   ships the CVE-2024-5206 fix the historical pin protected against). Pip resolves to 1.5.2.
# pyarrow (GHSA-rgxp-2hwp-jwgg): transitive dep; bump to >=23.0.1 to fix the CVE.
# cryptography: pinned transitive dep of azureml-acft-contrib-hf-nlp; fixes GHSA-g6cj-pr64-35w5.
# Remove pip's informational vendored-version manifest after installation; VCM
# reports its stale setuptools/msgpack entries even though the installed packages
# are patched by the constraints above.
RUN pip install --no-cache-dir -r requirements.txt 'aiohttp>=3.14.0' 'requests>=2.33.0' 'setuptools>=83.0.0' 'msgpack>=1.2.1' 'cryptography>=50.0.0' 'pyarrow>=23.0.1' && \
    find $AZUREML_CONDA_ENVIRONMENT_PATH \( -path '*/pip/_vendor/vendor.txt' -o -path '*/pip/_vendor/bom.cdx.json' \) -delete && \
    rm -rf /root/.cache/pip

# The below file is required for baking the code into the environment
COPY data_import_run.py /azureml/data_import/run.py
