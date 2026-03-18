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
RUN apt-get update && \
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

# Upgrade sudo to patch known vulnerability
RUN apt-get update && \
    apt-get install -y sudo=1.9.9-1ubuntu2.5 && \
    apt-mark hold sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# try updating pip for base and ptca env using conda
RUN conda install pip -n base -y 
RUN conda install pip -n ptca -y 

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
RUN pip install pyarrow==14.0.2
RUN pip install --upgrade torch==2.8.0 torchvision==0.23.0
RUN pip install --upgrade urllib3>=2.6.3

# Update conda base and ptca envs
RUN /opt/conda/bin/pip install --upgrade requests urllib3 
RUN /opt/conda/envs/ptca/bin/pip install --upgrade torch==2.8.0 torchvision==0.23.0 
RUN /opt/conda/envs/ptca/bin/pip install --upgrade urllib3>=2.6.3 filelock>=3.20.3

# Patch pillow vulnerability
RUN pip install --upgrade pillow==12.1.1
RUN /opt/conda/bin/pip install --upgrade pillow==12.1.1 
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pillow==12.1.1 

# Fix security vulnerabilities
RUN pip install --upgrade 'aiohttp>=3.13.3' 'azure-core>=1.38.0' 'bokeh>=3.8.2' 'distributed>=2026.1.0' 'jaraco.context>=6.1.0' 'protobuf>=6.33.5' 'pyasn1>=0.6.2' 'pip>=26.0' 'cryptography>=46.0.5' 'tornado>=6.5.5'

# Patch Ubuntu security vulnerabilities
# USN-7946-1 (GnuPG), USN-7951-1/USN-8018-1/USN-8018-2 (Python),
# USN-7954-1 (Libtasn1), USN-7963-1/USN-8039-1 (libpng),
# USN-7974-1 (libxml2), USN-7980-1 (OpenSSL), USN-8005-1 (glibc),
# USN-8017-1 (GLib), USN-8022-1 (Expat), USN-8025-1 (.NET),
# USN-8038-1 (nginx), USN-8043-1 (GnuTLS), USN-8044-1 (alsa-lib),
# USN-8062-1/USN-8084-1 (curl)
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        gnupg \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        dirmngr \
        python3.10 \
        python3.10-minimal \
        libpython3.10-stdlib \
        libpython3.10-minimal \
        libtasn1-6 \
        libpng16-16 \
        libxml2 \
        openssl \
        libssl3 \
        libc6 \
        libc-bin \
        libc6-dev \
        libc-dev-bin \
        libglib2.0-0 \
        libexpat1 \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        nginx-light \
        libgnutls30 \
        libasound2 \
        libcurl4 \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
# dummy number to change when needing to force rebuild without changing the definition: 1
