FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl-gpu
# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV ENABLE_METADATA=false

RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
RUN apt-get update && echo 'success updated apt-get!'
RUN apt-get install -y --no-install-recommends cmake libboost-dev libboost-system-dev libboost-filesystem-dev
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.8 pip=20.2.4 -y

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
    pip install --upgrade --force-reinstall xgboost==1.3.3 pystan==2.19.1.1 cmdstanpy==0.9.5 pandas==1.1.5"

# Create conda environment
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    numpy~=1.21.6 \
    scikit-learn=0.22.1 \
    pandas~=1.1.5 \
    holidays=0.10.3 \
    setuptools-git \
    'psutil>5.0.0,<6.0.0' \
    cudatoolkit=10.0.130 \
    -c anaconda -c conda-forge -c pytorch

# Install pip dependencies
RUN pip install azureml-core=={{latest-image-tag}}
                azureml-mlflow=={{latest-image-tag}}
                azureml-pipeline-core=={{latest-image-tag}}
                azureml-telemetry=={{latest-image-tag}}
                azureml-defaults=={{latest-image-tag}}
                azureml-interpret=={{latest-image-tag}}
                azureml-responsibleai=={{latest-image-tag}}
                azureml-automl-core=={{latest-image-tag}}
                azureml-automl-runtime=={{latest-image-tag}}
                azureml-train-automl-client=={{latest-image-tag}}
                azureml-train-automl-runtime=={{latest-image-tag}}
                azureml-dataset-runtime=={{latest-image-tag}}
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'inference-schema' \
                'py-cpuinfo==5.0.0' \
                'fbprophet==0.7.1' \
                'pystan==2.19.1.1' \
                'boto3==1.15.18' \
                'botocore==1.18.18'