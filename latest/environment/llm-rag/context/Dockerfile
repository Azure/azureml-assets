FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:1a43cb2d614666fcd532d78abe94afc741f518e90c94b08da12688a8eeea1677

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