FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.2

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
RUN /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade \
        'pip==26.1.1' \
        'python-dotenv>=1.2.2' \
        'urllib3>=2.7.0' \
        'idna>=3.15' \
        'click>=8.3.3' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info \
           /opt/conda/lib/python3.13/site-packages/python__dotenv-1.2.1.dist-info \
           /opt/conda/lib/python3.13/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python3.13/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python3.13/site-packages/click-8.2*.dist-info

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@a1ce2f9
