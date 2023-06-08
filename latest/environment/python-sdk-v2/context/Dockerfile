FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:60b340c2b7b83db890391136ca43f2f9a7f52c220c9e896b15790d6e43278d03

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/python-sdk-v2
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
