FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:20250414.v1

USER root

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml-automl-dnn-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230306.v3 /var/mlflow_resources/mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true

RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
RUN apt-get update && echo 'success updated apt-get!'
RUN apt-get install -y --no-install-recommends cmake libboost-dev libboost-system-dev libboost-filesystem-dev
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.9 pip=22.1.2 -y

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH

###############################
# Pre-Build LightGBM
###############################
RUN pip install --upgrade lightgbm==4.6.0
                
###############################
# Install GPU LightGBM and XgBoost
###############################
RUN pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3

# begin conda create
# Install conda packages
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    # begin conda dependencies
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    holidays=0.29 \
    setuptools-git \
    setuptools=72.1.0 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    joblib=1.2.0 \
    # end conda dependencies
    -c conda-forge -c anaconda

# Ensure future conda and pip calls are tied to this conda environment,
# and we don't just get lucky from the implicit PATH ordering.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]
# end conda create

# Conda installs from extra channels. Separate these to avoid solver OOMs.
RUN conda install pytorch=2.2.2 -c pytorch -c nvidia && conda clean -a -y
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.60.0 \
                azureml-pipeline-core==1.60.0 \
                azureml-telemetry==1.60.0 \
                azureml-defaults==1.60.0 \
                azureml-interpret==1.60.0 \
                azureml-responsibleai==1.60.0 \
                azureml-automl-core==1.60.0 \
                azureml-automl-runtime==1.60.0 \
                azureml-train-automl-client==1.60.0 \
                azureml-train-automl-runtime==1.60.0 \
                azureml-dataset-runtime==1.60.0 \
                azureml-mlflow==1.60.0.post1 \
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'azure-identity>=1.16.1' \
                'inference-schema' \
                'prophet==1.1.4' \
                'pytorch-transformers==1.0.0' \
                'spacy==3.7.4' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0' \
                'psutil>5.0.0,<6.0.0' \
                'requests>=2.31.0' \
                'certifi>=2023.07.22' \
                'cryptography>=42.0.5'
                # end pypi dependencies

RUN HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d

# end pip install

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
