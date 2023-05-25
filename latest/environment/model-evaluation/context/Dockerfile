FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:34cf59d95d14e145329e112f100b8a7748973d6a9b978f0f75920bd415bbaca6

WORKDIR /
ENV CONDA_PREFIX=/azureml-envs/model-evaluation
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX

COPY conda.yaml .
# Create conda environment
RUN conda env create -p $CONDA_PREFIX -f conda.yaml -q && \
    rm conda.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Prepend path to AzureML conda environment
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
