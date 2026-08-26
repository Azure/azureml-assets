FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
        curl \
        gzip \
        libcurl3-gnutls \
        libcurl4 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libnghttp2-14 \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libpam0g \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libsqlite3-0 \
        python3.10 \
        python3.10-minimal \
        tar \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ptca env (python 3.10): pip is a preinstalled install tool, not pulled by a
# Python parent package. Use conda-forge for the patched build and remove stale metadata.
# The existing pip in this env was installed by pip itself, so conda does not own its
# files and overlaying a conda pip on top leaves a mix of two pip trees behind
# ("ImportError: cannot import name 'get_runnable_pip' from 'pip._internal.utils.misc'").
# Remove the pip package directory first so conda lays down a coherent installation.
RUN rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-*.dist-info && \
    conda install -y -n ptca -c conda-forge pip==26.1.2 && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.1.1*.json && \
    conda clean -ay

# ptca env (python 3.10) pip-managed overrides. Parent packages in this env
# do not yet require the patched floors for these transitive dependencies:
#   urllib3: requests<3,>=1.26 and similar consumers.
#   idna: requests, anyio, httpx, yarl, etc.
#   aiohttp: no parent currently pins >=3.14.0 (GHSA-jg22-mg44-37j8 / GHSA-hg6j-4rv6-33pg).
#   starlette: no parent currently pins >=1.0.1 (GHSA-86qp-5c8j-p5mr).
#   pydantic-settings: pinned transitive dep of the base image, fixes GHSA-4xgf-cpjx-pc3j.
RUN /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade --progress-bar off \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'aiohttp>=3.14.0' \
        'starlette>=1.0.1' \
        'pydantic-settings>=2.14.2' && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/idna-3.11.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/aiohttp-3.13*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/starlette-1.0.0.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/pydantic_settings-2.14.1.dist-info

# ptca env (python 3.10): torch is inherited from the torch280 base image;
# pin the patched CUDA build for GHSA-rrmf-rvhw-rf47,
# and keep torchvision/torchaudio aligned with the torch ABI.
RUN /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade \
        --progress-bar off \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.13.0+cu126' \
        'torchvision==0.28.0+cu126' \
        'torchaudio==2.11.0+cu126' && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/torch-2.8*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torch-2.10*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchvision-0.23*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchvision-0.25*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchaudio-2.8*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchaudio-2.10*.dist-info

# Override vulnerable base-environment dependencies whose parent ranges permit
# older versions. Patch the inherited tooling directly in both conda environments.
# cryptography: pinned transitive dep of the base image, fixes GHSA-g6cj-pr64-35w5.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade --progress-bar off --no-binary=cffi \
        'cryptography==50.0.0' && \
    rm -rf /opt/conda/lib/python*/site-packages/cryptography-4[68]*.dist-info

RUN conda run -n base python -m pip install --no-cache-dir --upgrade --progress-bar off \
        'pip==26.1.2' \
        'setuptools>=83.0.0' \
        'py-rattler>=0.24.0' \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' \
        'aiohttp>=3.14.0' \
        'PyJWT>=2.13.0' \
        'pydantic-settings>=2.14.2' \
        'msgpack>=1.2.1' && \
    rm -rf /opt/conda/lib/python*/site-packages/pip-26.0*.dist-info \
           /opt/conda/lib/python*/site-packages/pip-26.1.1.dist-info \
           /opt/conda/lib/python*/site-packages/setuptools-8[12]*.dist-info \
           /opt/conda/lib/python*/site-packages/py__rattler-0.23*.dist-info \
           /opt/conda/lib/python*/site-packages/python__dotenv-1.2.1.dist-info \
           /opt/conda/lib/python*/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python*/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python*/site-packages/click-8.2*.dist-info \
           /opt/conda/lib/python*/site-packages/aiohttp-3.13*.dist-info \
           /opt/conda/lib/python*/site-packages/PyJWT-2.12*.dist-info \
           /opt/conda/lib/python*/site-packages/pydantic_settings-2.12*.dist-info \
           /opt/conda/lib/python*/site-packages/msgpack-1.1*.dist-info

RUN /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade --progress-bar off \
        'setuptools>=83.0.0' && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/setuptools-81*.dist-info

COPY requirements.txt .
RUN pip install --progress-bar off -r requirements.txt
RUN pip install --progress-bar off --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9
