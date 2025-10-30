FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202509.2


ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

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
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y

# vulnearbility fix
RUN pip install pyarrow==14.0.2
RUN pip install torch==2.8.0 torchvision==0.23.0
RUN opt/conda/envs/ptca/bin/pip install torch==2.8.0 torchvision==0.23.0 || true
RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install 2.8.0 torchvision==0.23.0 || true
RUN opt/conda/envs/ptca/bin/pip install --upgrade urllib3 || true
RUN opt/conda/bin/pip install --upgrade requests urllib3 || true
RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install --upgrade urllib3 || true
# Patch pillow vulnerability
RUN /azureml-envs/azureml-automl-dnn-vision-gpu/bin/pip install --upgrade pillow==11.3.0 || true
RUN /opt/conda/bin/pip install --upgrade pillow==11.3.0 || true
RUN /opt/conda/envs/ptca/bin/pip install --upgrade pillow==11.3.0 || true


ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
# dummy number to change when needing to force rebuild without changing the definition: 1
