FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:4505a963b2d34a42b5eaef48b8142ca98123d05d3a95eee3fe57551570a70e3b

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
