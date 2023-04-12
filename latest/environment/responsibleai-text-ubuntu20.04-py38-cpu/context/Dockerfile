FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:baf6420f6eac8aa72f2328792d4bf03231f5635aeb49fd402ea3f4ab31e1f1b7

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/responsibleai-text

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
RUN pip install 'azureml-dataset-runtime==1.50.0' \
                'azureml-core==1.50.0' \
                'azureml-mlflow==1.50.0' \
                'azureml-rai-utils==0.0.2'

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
