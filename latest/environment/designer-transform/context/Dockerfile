FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:42d89a0409f76aab4b23eb0d78a4917eb6ed9c9eeab44d5292e69650ae05e8ba

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/designer-transform
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

