FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/designer-transform
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# update to fix vulnerability
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl=7.81.0-1ubuntu1.20 && \
    linux-libc-dev=5.15.0-130.140 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

