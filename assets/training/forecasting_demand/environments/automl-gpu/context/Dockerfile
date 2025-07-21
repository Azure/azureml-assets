FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true
# Upgrade critical system and python packages
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        systemd=249.11-0ubuntu3.16 \
        systemd-sysv=249.11-0ubuntu3.16 \
        libudev1=249.11-0ubuntu3.16 \
        libpam-systemd=249.11-0ubuntu3.16 \
        systemd-timesyncd=249.11-0ubuntu3.16 \
        libsystemd0=249.11-0ubuntu3.16 \
        libnss-systemd=249.11-0ubuntu3.16 \
        libpython3.10-stdlib=3.10.12-1~22.04.10 \
        python3.10=3.10.12-1~22.04.10 \
        libpython3.10-minimal=3.10.12-1~22.04.10 \
        python3.10-minimal=3.10.12-1~22.04.10 \
        libpam0g=1.4.0-11ubuntu2.6 \
        libpam-modules-bin=1.4.0-11ubuntu2.6 \
        libpam-modules=1.4.0-11ubuntu2.6 \
        libpam-runtime=1.4.0-11ubuntu2.6 \
        libarchive13=3.6.0-1ubuntu1.5 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y
# dummy number to change when needing to force rebuild without changing the definition: 2

# Address vulnerabilities
RUN /opt/miniconda/bin/pip install --upgrade 'requests==2.32.4' 'urllib3==2.5.0' || true
RUN $AZUREML_CONDA_ENVIRONMENT_PATH/bin/pip install --upgrade 'urllib3==2.5.0' 'torch==2.7.1' 'torchvision==0.22.1' || true
# If ptca env exists (for miniconda setups)
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'urllib3==2.5.0' 'torch==2.7.1' 'torchvision==0.22.1' || true

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib
