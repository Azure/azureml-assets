FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20251110.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

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
        libarchive13 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y
# dummy number to change when needing to force rebuild without changing the definition: 2

# Address vulnerabilities
# RUN /opt/miniconda/bin/pip install --upgrade 'requests==2.32.4' 'urllib3==2.5.0' 'pillow==11.3.0' || true
RUN pip install --upgrade --no-cache-dir 'requests==2.32.5' 'urllib3==2.5.0' 'torch==2.8.0' 'torchvision==0.23.0' 'pillow==11.3.0' || true
# If ptca env exists (for miniconda setups)
# RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'urllib3==2.5.0' 'torch==2.8.0' 'torchvision==0.22.1' 'pillow==11.3.0' || true

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib
