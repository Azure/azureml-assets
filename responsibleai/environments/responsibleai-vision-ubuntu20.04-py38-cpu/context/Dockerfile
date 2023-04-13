FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

# Install OpenCV native dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get install -y python3-opencv

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-vision

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0' 'numpy<1.24.0'

# Install azureml packages
RUN pip install 'azureml-dataset-runtime=={{latest-pypi-version}}' \
                'azureml-core=={{latest-pypi-version}}' \
                'azureml-mlflow=={{latest-pypi-version}}' \
                'azureml-automl-dnn-vision=={{latest-pypi-version}}' \
                'azureml-rai-utils==0.0.2' \
                'https://publictestdatasets.blob.core.windows.net/packages/pypi/responsibleai-vision/responsibleai_vision-0.0.25-py3-none-any.whl'
# Todo: Need to install rai-vision-utils from wheels and update the raivisioninsights package wheel 

# Work around to resolve error: The user requested ml-wrappers~=0.4.2; interpret-community 0.27.0 depends on ml-wrappers==0.2.2
RUN pip install 'ml-wrappers~=0.4.2' \
                'shap==0.40.0' \
                'interpret-community==0.28.0' \
                'Pillow>=9.1.0'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
