FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1

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

# Install required PyPI packages.
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# torch is a direct training dependency; install CUDA 12.6 PyTorch wheels to
# override vulnerable torch 2.8.0+cu126 from the ACPT base image (GHSA-vgrw-7cvw-pwgx,
# GHSA-qfhq-4f3w-5fph). torchvision and torchaudio are upgraded with torch to
# keep the CUDA PyTorch stack compatible.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.10.0' \
        'torchvision==0.25.0' \
        'torchaudio==2.10.0'

# Base-env transitive dependency overrides. The base image is patched for some
# of these, but this image's dependency layer can resolve AzureML/MLflow
# transitive dependencies below the fixed floors.
# urllib3 2.6.3 is pulled in by requests 2.33.1 (`urllib3<3,>=1.26`);
# requests 2.34.2 still does not require urllib3>=2.7.0.
# idna 3.11 is required by requests 2.33.1, httpx 0.28.1, anyio 4.12.1, and
# yarl 1.23.0; their parent metadata still does not require idna>=3.15.
# click 8.2.1 is required by anaconda-cli-base 0.8.2 and typer 0.25.1; parent
# metadata only requires `click` or `click>=8.2.1`, not click>=8.3.3.
# python-dotenv 1.2.1 is required by anaconda-auth 0.14.4 and
# pydantic-settings; parents still use unpinned python-dotenv or
# python-dotenv>=0.21.0.
RUN conda install -y -n base -c conda-forge 'urllib3=2.7.0' 'idna=3.15' 'click=8.3.3' 'python-dotenv=1.2.2' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/urllib3-2.6*.dist-info \
           /opt/conda/lib/python3.13/site-packages/idna-3.11.dist-info \
           /opt/conda/lib/python3.13/site-packages/click-8.2*.dist-info \
           /opt/conda/lib/python3.13/site-packages/python_dotenv-1.2.1.dist-info \
           /opt/conda/lib/python3.13/site-packages/python-dotenv-1.2.1.dist-info && \
    conda clean -ay

# The conda-forge conda update can add optional conda-rattler-solver/py-rattler
# metadata. This image does not need that solver plugin after build-time package
# installation, so remove it to avoid carrying unused solver artifacts.
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

# pyarrow: pinned transitive dep in ptca; brought in by AzureML dependencies, fixes GHSA-rgxp-2hwp-jwgg.
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-537c-gmf6-5ccf.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade 'pyarrow>=23.0.1' 'cryptography==48.0.1'

# PyJWT: pinned transitive dep in base; brought in by base image, fixes GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4.
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-537c-gmf6-5ccf.
# pydantic-settings: pinned transitive dep of azureml-defaults -> azureml-inference-server-http, fixes GHSA-4xgf-cpjx-pc3j.
# msgpack: pinned transitive dep of azureml-mlflow -> mlflow-skinny, fixes GHSA-6v7p-g79w-8964.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.13.0' 'cryptography==48.0.1' 'pydantic-settings>=2.14.2' 'msgpack>=1.2.1' && \
    rm -rf /opt/conda/lib/python3.13/site-packages/pydantic_settings-2.12*.dist-info \
           /opt/conda/lib/python3.13/site-packages/pydantic__settings-2.12*.dist-info \
           /opt/conda/lib/python3.13/site-packages/msgpack-1.1*.dist-info

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
