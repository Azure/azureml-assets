FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:{{latest-image-tag}}

WORKDIR /

COPY conda_dependencies.yaml conda_dependencies.yaml

ENV CONDA_PREFIX /azureml-envs/tensorflow-2.8
ENV CONDA_DEFAULT_ENV /azureml-envs/tensorflow-2.8

# Create conda environment
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml

# Prepend path to AzureML conda environment
ENV PATH $CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
