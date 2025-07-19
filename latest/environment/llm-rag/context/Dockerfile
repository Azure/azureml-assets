FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20250717.v1

# Set the shared environment path
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/rag

# Prepend the path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y