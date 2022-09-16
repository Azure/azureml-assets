FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:4a02c34ae3eded54b85ec52da5c0f2c7b351f801fd129c875176d1bb1a85aded

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.1
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
