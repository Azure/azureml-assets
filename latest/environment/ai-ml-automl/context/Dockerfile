FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20251110.v1


ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

# System package security upgrades
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        libpam0g \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip=25.3 \
    py-cpuinfo=5.0.0\
    joblib=1.2.0 \
    setuptools-git \
    setuptools=78.1.1 \
    'psutil>5.0.0,<6.0.0' \
    pytorch=2.8.0 \
    # end conda dependencies
    -c conda-forge -c pytorch -c anaconda -c nvidia && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.61.0 \
                azureml-mlflow==1.61.0 \
                azureml-pipeline-core==1.61.0 \
                azureml-telemetry==1.61.0 \
                azureml-interpret==1.61.0 \
                azureml-responsibleai==1.61.0 \
                azureml-automl-core==1.61.0 \
                azureml-automl-runtime==1.61.0 \
                azureml-train-automl-client==1.61.0 \
                azureml-train-automl-runtime==1.61.0 \
                azureml-train-automl==1.61.0 \
                azureml-dataset-runtime==1.61.0 \
                azureml-defaults==1.61.0 \
                # TODO: replace the hard coded above by a referenceto azureml-train-automl[tabular]
                'mlflow-skinny==2.15.1' \
                'xgboost==1.5.2' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'pillow==12.0.0'
                # end pypi dependencies
# end pip install

# begin pip ad-hoc
# Install pip ad-hoc dependencies for security updates
RUN pip install --upgrade 'pip>=25.3' 'starlette>=0.49.1' 'pyarrow==14.0.2' 'urllib3==2.5.0'
# end pip ad-hoc
