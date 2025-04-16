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
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.9 pip=22.1.2 conda-forge::tzdata -y

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH

###############################
# Pre-Build LightGBM
###############################
RUN cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
    git clone --recursive --branch v3.2.1 --depth 1 https://github.com/microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make -j4
                
###############################
# Install GPU LightGBM and XgBoost
###############################
RUN /bin/bash -c "cd /usr/local/src/lightgbm/LightGBM/python-package && python setup.py install --precompile && \
    pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3"

# begin conda create
# Create conda environment
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    # begin conda dependencies
    numpy~=1.23.5 \
    scikit-learn=1.5.1 \
    holidays=0.29 \
    setuptools-git \
    setuptools=72.1.0 \
    wheel=0.44.0 \
    scipy=1.10.1 \
    'psutil>5.0.0,<6.0.0' \
    cudatoolkit=10.0.130 \
    # end conda dependencies
    -c conda-forge -c pytorch -c anaconda
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core==1.60.0 \
                azureml-mlflow==1.60.0 \
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
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'azure-identity>=1.16.1' \
                'inference-schema' \
                'py-cpuinfo==5.0.0' \
                'prophet==1.1.4' \
                'cryptography>=42.0.5' \
                'requests>=2.31.0' \
                'certifi>=2023.07.22'
                # end pypi dependencies
# end pip install

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
