FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20250828.v1

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/designer-r
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

USER root

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y 

RUN conda run -p $CONDA_PREFIX conda remove -p $CONDA_PREFIX --yes r-bslib
RUN rm -rf /opt/miniconda/pkgs/

