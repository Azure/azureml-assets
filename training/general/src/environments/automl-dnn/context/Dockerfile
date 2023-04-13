FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl-dnn
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV ENABLE_METADATA=false

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=20.2.4 \
    numpy~=1.21.6 \
    py-cpuinfo=5.0.0 \
    boto3=1.15.18 \
    botocore=1.18.18 \
    joblib=1.2.0 \
    cloudpickle=1.6.0 \
    scikit-learn=0.22.1 \
    pandas~=1.1.5 \
    py-xgboost=1.3.3 \
    holidays=0.10.3 \
    setuptools-git \
    pytorch=1.4.0 \
    cudatoolkit=10.0.130 \
    'psutil>5.0.0,<6.0.0' \
    GitPython=3.1.30 \
    -c anaconda -c conda-forge -c pytorch && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y

# Install pip dependencies
RUN pip install azureml-core=={{latest-pypi-version}}
                azureml-mlflow=={{latest-pypi-version}}
                azureml-pipeline-core=={{latest-pypi-version}}
                azureml-telemetry=={{latest-pypi-version}}
                azureml-interpret=={{latest-pypi-version}}
                azureml-responsibleai=={{latest-pypi-version}}
                azureml-automl-core=={{latest-pypi-version}}
                azureml-automl-runtime=={{latest-pypi-version}}
                azureml-train-automl-client=={{latest-pypi-version}}
                azureml-train-automl-runtime=={{latest-pypi-version}}
                azureml-dataset-runtime=={{latest-pypi-version}}
                azureml-defaults=={{latest-pypi-version}}
                'inference-schema' \
                'fbprophet==0.7.1' \
                'pystan==2.19.1.1' \
                'mltable>=1.0.0' \
                'pytorch-transformers==1.0.0' \
                'spacy==2.1.8' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-2.1.0.tar.gz'