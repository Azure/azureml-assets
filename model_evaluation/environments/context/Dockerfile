FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20220516.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/model-eval-env
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH


# Install conda dependencies
RUN conda install scikit-learn==0.24.1
# Install pip dependencies
RUN pip install 'matplotlib>=3.3,<3.4' \
                'protobuf<4.0.0' \
                'psutil>=5.8,<5.9' \
                'tqdm>=4.59,<4.60' \
                'pandas>=1.1,<1.2' \
                'ipykernel~=6.0' \
                'azureml-core==1.43.0' \
                'azureml-defaults==1.43.0' \
                'azureml-mlflow==1.43.0' \
                'azureml-telemetry==1.43.0' \
                'azureml-automl-core==1.43.0' \
                'azureml-automl-runtime==1.43.0' \
                'xgboost==1.6.1' \
                'statsmodels==0.11.0' \
                'lightgbm==3.2.1' \
                'msrest==0.6.21' \
                'torch' \
                'torchaudio' \
                'torchvision' \
                'transformers' \
                'tokenizers' \
                'datasets' \
                'mltable==0.1.0b3'


RUN pip install --extra-index-url https://azuremlsdktestpypi.azureedge.net/Create-Dev-Index/74090581 azureml-metrics
RUN pip install --extra-index-url https://azuremlsdktestpypi.azureedge.net/Create-Dev-Index/73836732 azureml-evaluate-mlflow

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH