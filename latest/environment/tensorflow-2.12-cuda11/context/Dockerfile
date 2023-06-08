FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04@sha256:dac0bfa3452e0e860b92a6ca453d543857d9faf83da06521b1bebd8a3331d43e

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/tensorflow-2.12-cuda11
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN HOROVOD_WITH_TENSORFLOW=1 conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

