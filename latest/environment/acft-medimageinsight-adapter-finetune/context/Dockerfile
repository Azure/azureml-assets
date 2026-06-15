FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.2

USER root

RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install --only-upgrade \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        libgcrypt20 \
        liblzma5 \
        xz-utils \
 || true \
 && apt-get -y install --no-install-recommends unzip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip is installed directly as the Python installer in both conda envs, so
# there is no parent package to upgrade. Base tag biweekly.202605.2 still ships
# pip 26.0.1 in base and ptca; conda-forge has patched 26.1.1 as of
# 2026-05-25, so keep this direct override before requirements installation.
RUN conda install -y -n base -c conda-forge pip==26.1.1 && \
    conda install -y -n ptca -c conda-forge pip==26.1.1 && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/pip-26.0*.dist-info && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json && \
    conda clean -ay

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Base-env transitive dependency overrides. Programmatic checks against
# biweekly.202605.2 showed ptca already has urllib3 2.7.0 and idna 3.15, and
# the requirements resolver now installs patched pyasn1, Mako, and
# python-dotenv in ptca without direct overrides.
#
# urllib3 2.6.3 is pulled in by requests 2.33.1 (`urllib3<3,>=1.26`);
# latest requests 2.34.2 still does not require urllib3>=2.7.0.
#
# idna 3.11 is required by requests 2.33.1, httpx 0.28.1, anyio 4.12.1, and
# yarl 1.23.0; their latest releases as of 2026-05-25 still do not require
# idna>=3.15.
#
# click 8.2.1 is required by anaconda-cli-base 0.8.2 and typer 0.25.1; latest
# parent metadata only requires `click` or `click>=8.2.1`, not click>=8.3.3.
#
# python-dotenv 1.2.1 is required by anaconda-auth 0.14.4 and
# pydantic-settings 2.12.0; latest parents still use unpinned python-dotenv or
# python-dotenv>=0.21.0, so upgrade the base-env transitive package directly.
RUN conda install -y -n base -c conda-forge 'urllib3=2.7.0' 'idna=3.15' 'click=8.3.3' 'python-dotenv=1.2.2' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python3.13/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python3.13/site-packages/click-8.2*.dist-info \
           /opt/conda/lib/python3.13/site-packages/python_dotenv-1.2.1.dist-info \
           /opt/conda/lib/python3.13/site-packages/python-dotenv-1.2.1.dist-info && \
    conda clean -ay

# The conda-forge conda update pulled by the conda installs above adds
# conda-rattler-solver 0.1.0 -> py-rattler 0.23.2. The current py-rattler
# release is still 0.23.2 as of 2026-05-25 and its rattler.abi3.so embeds
# vulnerable Rust crates, so no fixed parent package is available yet. This
# optional solver plugin is not needed after build-time package installation.
RUN /opt/conda/bin/pip uninstall -y py-rattler conda-rattler-solver && \
    rm -rf /opt/conda/lib/python3.13/site-packages/rattler \
           /opt/conda/lib/python3.13/site-packages/py_rattler* \
           /opt/conda/lib/python3.13/site-packages/py-rattler* \
           /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* && \
    rm -f /opt/conda/conda-meta/py-rattler-*.json \
          /opt/conda/conda-meta/conda-rattler-solver-*.json

# Upgrade aiohttp to patched version in both envs (GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' && \
    conda run -n ptca python -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
