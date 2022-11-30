FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:ffc4b931ad29c78fd6a0aef66f35eb110b9a5add2f10f4c17be9dd50ad3524a8

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

RUN pip install --pre 'azure-ai-ml'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
