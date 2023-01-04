FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.7.9 \
    pip=20.2.4 \
    numpy~=1.21.6 \
    py-cpuinfo=5.0.0 \
    click<8.1.0 \
    pystan=2.19.1.1 \
    boto3=1.15.18 \
    botocore=1.18.18 \
    joblib=0.14.1 \
    cloudpickle=1.6.0 \
    scikit-learn=0.22.1 \
    pandas~=1.1.5 \
    py-xgboost=1.3.3 \
    fbprophet=0.7.1 \
    holidays=0.10.3 \
    setuptools-git \
    psutil>5.0.0,<6.0.0 \
    -c conda-forge -c pytorch -c anaconda

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'azureml-core' \
                'azureml-mlflow' \
                'azureml-pipeline-core' \
                'azureml-telemetry' \
                'azureml-interpret' \
                'azureml-responsibleai' \
                'azureml-automl-core' \
                'azureml-automl-runtime' \
                'azureml-train-automl-client' \
                'azureml-train-automl-runtime' \
                'azureml-dataset-runtime' \
                'azureml-defaults' \
                'inference-schema' \
                'mpi4py==3.1.3'