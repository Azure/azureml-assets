FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:{{latest-image-tag}}


ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
# Security fixes: USN-8005-1 (glibc), USN-8018-1 (python), USN-8017-1 (glib),
# USN-8022-1 (expat), USN-8025-1 (.NET), USN-8039-1 (libpng), USN-8038-1 (nginx),
# USN-8043-1 (gnutls), USN-8044-1 (alsa-lib), USN-8062-1 (curl), USN-7946-1 (gnupg),
# USN-7951-1 (python), USN-7954-1 (libtasn1), USN-7963-1 (libpng), USN-7974-1 (libxml2),
# USN-7980-1 (openssl)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        libc-bin \
        dpkg-dev \
        libssl-dev \
        dpkg \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        binutils \
        binutils-common \
        binutils-x86-64-linux-gnu \
        libbinutils \
        libctf0 \
        libctf-nobfd0 \
        libc6 \
        libc6-dev \
        libc-dev-bin \
        libssh-4 \
        libxml2 \
        linux-libc-dev \
        linux-headers-generic \
        locales \
        openssl \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

# Upgrade sudo
RUN apt-get update && \
    apt-get install -y sudo && \
    apt-mark hold sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# try updating pip for base and ptca env using conda
RUN conda install pip -n base -y || true
RUN conda install pip -n ptca -y || true

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda clean -afy

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-automl-dnn-vision=={{latest-pypi-version}}

# vulnearbility fix
RUN pip install --no-cache-dir pyarrow==14.0.2
RUN pip install --upgrade --no-cache-dir torch==2.8.0 torchvision==0.23.0

# Security vulnerability fixes for main conda env:
# GHSA-58pv-8j8x-9vj2 jaraco.context (vendored in setuptools)
# GHSA-8rrh-rw8j-w5fx wheel (vendored in setuptools + direct)
# GHSA-r6ph-v2qm-q3c2 cryptography
# GHSA-cfh3-3jmp-rvhc pillow
# GHSA-c336-7962-wfj2 distributed (transitive via dask)
# GHSA-793v-589g-574v bokeh
# GHSA-6vgw-5pg2-w6jp pip
# GHSA-38jv-5279-wg99 urllib3
RUN pip install --upgrade --no-cache-dir \
    'setuptools>=82.0.1' \
    'wheel>=0.46.2' \
    'cryptography>=46.0.5' \
    'pillow>=12.1.1' \
    'distributed>=2026.1.0' \
    'dask>=2026.1.0' \
    'bokeh>=3.8.2' \
    'pip>=26.0' \
    'urllib3>=2.6.3'

# Security fixes for base conda env:
# GHSA-r6ph-v2qm-q3c2 cryptography, GHSA-8rrh-rw8j-w5fx wheel,
# GHSA-cfh3-3jmp-rvhc pillow, GHSA-69f9/6mq8/mqqc/fh55/6jhg/jj3x/g84x/54jq aiohttp
# GHSA-58pv-8j8x-9vj2 jaraco.context (vendored in setuptools)
RUN /opt/conda/bin/pip install --upgrade --no-cache-dir \
    'cryptography>=46.0.5' \
    'wheel>=0.46.2' \
    'pillow>=12.1.1' \
    'aiohttp>=3.13.3' \
    'urllib3>=2.6.3' || true
RUN /opt/conda/bin/python -m pip install --upgrade --no-cache-dir 'setuptools>=82.0.1' || true

# Security fixes for ptca conda env:
# GHSA-58pv-8j8x-9vj2 jaraco.context (vendored in setuptools),
# GHSA-8rrh-rw8j-w5fx wheel, GHSA-cfh3-3jmp-rvhc pillow
RUN /opt/conda/envs/ptca/bin/pip install --upgrade --no-cache-dir \
    'wheel>=0.46.2' \
    'pillow>=12.1.1' \
    'urllib3>=2.6.3' \
    'filelock>=3.20.3' || true
RUN /opt/conda/envs/ptca/bin/python -m pip install --upgrade --no-cache-dir 'setuptools>=82.0.1' || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade --no-cache-dir torch==2.8.0 torchvision==0.23.0 || true


ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
# dummy number to change when needing to force rebuild without changing the definition: 1
