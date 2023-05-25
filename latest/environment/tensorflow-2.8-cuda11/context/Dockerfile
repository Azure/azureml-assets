FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04@sha256:d555a351750531459edeff521d65165bc360e345faf915e3d2689dce668b74b1

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/tensorflow-2.8
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

