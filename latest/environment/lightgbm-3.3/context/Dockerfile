FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:d87e60a40761fd1bb9efa97d94a9ec52b62a9e7081fd2acd5c7d94ba2852d60a

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/lightgbm-3.3
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
