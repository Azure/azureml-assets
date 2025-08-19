# PTCA Image (enabling Singularity Support)
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202508.1

RUN apt-get update && apt-get -y upgrade

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/model-evaluation
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

COPY conda.yaml .
# Create conda environment
RUN conda env create -p $CONDA_PREFIX -f conda.yaml -q && \
    rm conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Temp fix for env vuln
RUN pip install mlflow==3.3.0
RUN pip install transformers==4.55.2
