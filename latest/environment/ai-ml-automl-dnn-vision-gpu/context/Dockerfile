FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260:biweekly.202506.3

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
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# Create conda environment
# begin conda create
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.9 \
    # begin conda dependencies
    pip=21.3.1 \
    numpy~=1.23.5 \
    libffi=3.3 \
    pycocotools=2.0.4 \
    shap=0.39.0 \
    llvmlite=0.39.1 \
    scipy=1.10.1 \
    setuptools=78.1.1 \
    wheel=0.44.0 \
    tbb=2021.1.1 \
    # end conda dependencies
    -c conda-forge -c cerebis && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install

# Install pip dependencies
RUN pip install \
                 # begin pypi dependencies
                azureml-mlflow==1.60.0.post1 \
                azureml-dataset-runtime==1.60.0 \
                azureml-telemetry==1.60.0 \
                azureml-responsibleai==1.60.0 \
                azureml-automl-core==1.60.0 \
                azureml-automl-runtime==1.60.0 \
                azureml-train-automl-client==1.60.0 \
                azureml-defaults==1.60.0 \
                azureml-interpret==1.60.0 \
                azureml-train-automl-runtime==1.60.0 \
                azureml-automl-dnn-vision==1.60.0 \
                'azureml-dataprep>=2.24.4' \
                'azure-identity>=1.16.1'
                # end pypi dependencies

# Update cryptography and pyarow for fixing vulnerability. Doing it  separately from pip install to avoid conflict with other packages
RUN pip install cryptography>=42.0.5 \
                pyarrow==14.0.2 \
                aiohttp>=3.9.4

# end pip install
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
