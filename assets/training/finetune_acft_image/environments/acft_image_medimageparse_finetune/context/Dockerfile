FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ptca env (python 3.10): pip is a preinstalled install tool, not pulled by a
# Python parent package. conda defaults still resolves the vulnerable 26.0.1
# build, so use conda-forge for the patched build and remove stale metadata.
RUN conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# ptca env (python 3.10) pip-managed overrides. Parent packages in this env
# do not yet require the patched floors for these transitive dependencies:
#   urllib3: requests<3,>=1.26 and similar consumers.
#   idna: requests, anyio, httpx, yarl, etc.
#   aiohttp: no parent currently pins >=3.14.0 (GHSA-jg22-mg44-37j8 / GHSA-hg6j-4rv6-33pg).
#   starlette: no parent currently pins >=1.0.1 (GHSA-86qp-5c8j-p5mr).
RUN /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'aiohttp>=3.14.0' \
        'starlette>=1.0.1' && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/idna-3.11.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/aiohttp-3.13*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/starlette-1.0.0.dist-info

# Base conda env (python 3.13) leaf overrides. Programmatic metadata checks on
# 2026-05-25 found no available parent package that tightens these vulnerable
# transitive dependencies to the fixed floors:
#   pip: preinstalled install tool; no Python parent package.
#   python-dotenv: anaconda-auth (unversioned) and pydantic-settings>=0.21.0.
#   urllib3: requests<3,>=1.26 and anaconda-client>=1.26.4.
#   idna: requests<4,>=2.5, anyio>=2.8, httpx, and yarl>=2.0.
#   click: anaconda-cli-base, typer>=8.2.1, and optional httpx/python-dotenv
#   extras. Patch directly until those parents or the base env require fixed
#   versions.
#   py-rattler: pinned transitive dep of base conda/rattler tooling; brought in
#   by the base image, fixes GHSA-q53q-5r4j-5729.
#   aiohttp: no parent currently pins >=3.14.0 (GHSA-jg22-mg44-37j8 / GHSA-hg6j-4rv6-33pg).
#   PyJWT: no parent currently pins >=2.13.0 (GHSA-jq35-7prp-9v3f / GHSA-993g-76c3-p5m4).
#   cryptography: pinned transitive dep of the base Python stack; brought in by
#   the base image, fixes GHSA-537c-gmf6-5ccf.
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade --no-binary=cffi \
        'cryptography==48.0.1' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/cryptography-46*.dist-info

RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade \
        'pip==26.1.1' \
        'py-rattler>=0.24.0' \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' \
        'aiohttp>=3.14.0' \
        'PyJWT>=2.13.0' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info \
           /opt/conda/lib/python3.13/site-packages/py__rattler-0.23*.dist-info \
           /opt/conda/lib/python3.13/site-packages/python__dotenv-1.2.1.dist-info \
           /opt/conda/lib/python3.13/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python3.13/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python3.13/site-packages/click-8.2*.dist-info \
           /opt/conda/lib/python3.13/site-packages/aiohttp-3.13*.dist-info \
           /opt/conda/lib/python3.13/site-packages/PyJWT-2.12*.dist-info

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9
