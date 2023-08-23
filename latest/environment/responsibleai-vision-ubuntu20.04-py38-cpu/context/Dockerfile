FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230808.v1

# Install OpenCV native dependencies
RUN apt-get update
RUN apt-get install -y python3-opencv

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-vision

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN pip install --pre 'azure-ai-ml' 'azure-storage-blob<=12.13.0'

# Install azureml packages
RUN pip install 'azureml-core==1.53.0' \
                'azureml-mlflow==1.53.0' \
                'azureml-rai-utils==0.0.3'

RUN pip install 'azureml-dataset-runtime==1.53.0' \
                'azureml-automl-dnn-vision==1.53.0'

RUN pip install 'shap==0.40.0' \
                'interpret-community==0.29.0' \
                'Pillow>=9.1.0' \
                'Werkzeug==2.2.3' \
                'setuptools>=65.5.1' \
                'numpy==1.22.0' \
                'scipy==1.10.0' \
                'statsmodels==0.14.0'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
