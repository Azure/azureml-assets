FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:34cf59d95d14e145329e112f100b8a7748973d6a9b978f0f75920bd415bbaca6

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
