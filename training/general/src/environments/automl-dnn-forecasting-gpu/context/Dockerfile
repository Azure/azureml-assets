FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl-dnn-forecasting-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV ENABLE_METADATA=false

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=22.1.2 \
    pytorch=1.9.0 \
    numpy~=1.21.6 \
    scikit-learn=0.22.1 \
    pandas~=1.1.5 \
    py-xgboost=1.3.3 \
    cudatoolkit=11.1 \
    holidays=0.10.3 \
    'psutil>=5.2.2,<6.0.0' \
    GitPython=3.1.30 \
    'tensorboard>=1.14.0,<=2.2.2' \
    pip \
    tqdm \
    -c anaconda -c conda-forge -c pytorch && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH && \
    conda clean -a -y

# Install pip dependencies
RUN pip install 'inference-schema' \
                'horovod==0.22.1' \
                azureml-core=={{latest-pypi-version}}
                azureml-mlflow=={{latest-pypi-version}}
                azureml-defaults=={{latest-pypi-version}}
                azureml-telemetry=={{latest-pypi-version}}
                azureml-interpret=={{latest-pypi-version}}
                azureml-responsibleai=={{latest-pypi-version}}
                azureml-automl-core=={{latest-pypi-version}}
                azureml-automl-runtime=={{latest-pypi-version}}
                azureml-train-automl-client=={{latest-pypi-version}}
                azureml-train-automl-runtime=={{latest-pypi-version}}
                azureml-dataset-runtime=={{latest-pypi-version}}
                azureml-contrib-dataset=={{latest-pypi-version}}
                azureml-train-automl=={{latest-pypi-version}}
                azureml-contrib-automl-dnn-forecasting=={{latest-pypi-version}}
                'spacy==2.1.8' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-2.1.0.tar.gz' \
                'py-cpuinfo==5.0.0'