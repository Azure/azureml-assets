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
RUN pip install 'azureml-dataset-runtime==1.48.0' \
                'azureml-core==1.48.0' \
                'azureml-mlflow==1.48.0' \
                'azureml-automl-dnn-vision==1.48.0' \
                'https://publictestdatasets.blob.core.windows.net/packages/pypi/azureml-model-serializer/azureml_model_serializer-0.0.16-py3-none-any.whl'

# Work around to resolve error: The user requested ml-wrappers~=0.4.2; interpret-community 0.27.0 depends on ml-wrappers==0.2.2
RUN pip install 'ml-wrappers~=0.4.2' \
                'shap==0.40.0' \
                'interpret-community==0.28.0' \
                'Pillow>=9.1.0'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
