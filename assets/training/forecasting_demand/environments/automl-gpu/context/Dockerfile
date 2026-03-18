FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true
# Upgrade critical system and python packages
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        systemd \
        systemd-sysv \
        libudev1 \
        libpam-systemd \
        systemd-timesyncd \
        libsystemd0 \
        libnss-systemd \
        libpython3.10-stdlib \
        python3.10 \
        libpython3.10-minimal \
        python3.10-minimal \
        libpam0g \
        libpam-modules-bin \
        libpam-modules \
        libpam-runtime \
        libarchive13 \
        # USN-7942-1, USN-8017-1 (GLib)
        libglib2.0-0 \
        # USN-7946-1 (GnuPG)
        gnupg2 \
        # USN-7954-1 (Libtasn1)
        libtasn1-6 \
        # USN-7974-1 (libxml2)
        libxml2 \
        # USN-7980-1 (OpenSSL)
        openssl libssl3 \
        # USN-8005-1 (GNU C Library)
        libc6 libc-bin \
        # USN-8022-1 (Expat)
        libexpat1 \
        # USN-8043-1 (GnuTLS)
        libgnutls30 \
        # USN-8062-1, USN-8084-1 (curl)
        curl libcurl4 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# TODO: Patch needed for Ubuntu Security Notification for nginx Vulnerability (USN-8038-1) - nginx likely not installed in this image

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y
# dummy number to change when needing to force rebuild without changing the definition: 2

RUN pip install 'cryptography>=46.0.5'
# Security vulnerability overrides for the conda environment
# GHSA-793v-589g-574v (bokeh), GHSA-r6ph-v2qm-q3c2 (cryptography), GHSA-c336-7962-wfj2 (distributed)
# GHSA-38jv-5279-wg99 (urllib3), GHSA-87hc-h4r5-73f7 (Werkzeug), GHSA-qmgc-5h2g-mvrw (filelock)
# GHSA-jm66-cg57-jjv5 (azure-core), GHSA-58pv-8j8x-9vj2 (jaraco.context), GHSA-63vm-454h-vhhq (pyasn1)
# GHSA-7gcm-g887-7qv7 (protobuf), GHSA-cfh3-3jmp-rvhc (pillow), GHSA-78cv-mqj4-43f7 (tornado)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'bokeh>=3.8.2' \
    'cryptography>=46.0.5' \
    'setuptools>=79.0.0' \
    'distributed>=2026.1.0' \
    'urllib3>=2.6.3' \
    'Werkzeug>=3.1.5' \
    'filelock>=3.20.3' \
    'azure-core>=1.38.0' \
    'jaraco.context>=6.1.0' \
    'pyasn1>=0.6.2' \
    'protobuf>=5.29.6' \
    'pillow>=12.1.1' \
    'tornado>=6.5.5'

# Clean conda pkgs cache to remove stale vendored copies
RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib
