FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:2a7a3804e0b071870e78304aa13dab15c2ece3120f781a99d52f24ff6b71dea6

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
