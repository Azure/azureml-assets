FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20220516.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/model-evaluation

COPY conda.yaml .
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda.yaml -q && \
    rm conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install extra dependencies
RUN pip install --extra-index-url https://azuremlsdktestpypi.azureedge.net/Create-Dev-Index/74090581 azureml-metrics
RUN pip install --extra-index-url https://azuremlsdktestpypi.azureedge.net/Create-Dev-Index/73836732 azureml-evaluate-mlflow

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH