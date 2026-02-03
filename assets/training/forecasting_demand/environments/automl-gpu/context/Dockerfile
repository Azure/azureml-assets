FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

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
        libglib2.0-dev \
        libglib2.0-data \
        gnupg-utils \
        gpg-wks-server \
        gnupg2 \
        gpg-wks-client \
        gnupg \
        gpgconf \
        gpgsm \
        gpgv \
        gpg \
        dirmngr \
        gpg-agent \
        gnupg-l10n \
        libarchive13 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update base conda environment with dependencies
COPY conda_dependencies.yaml .
RUN conda env update --name base -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run --name base pip cache purge && \
    # dependency on the azureml-train-automl-runtime
    conda run --name base pip install bokeh==3.8.2 && \
    conda clean -a -y
