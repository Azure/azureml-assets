FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04@sha256:baf6420f6eac8aa72f2328792d4bf03231f5635aeb49fd402ea3f4ab31e1f1b7

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

RUN pip install "azureml-evaluate-mlflow==0.0.4" --upgrade
RUN pip install "azureml-metrics==0.0.4.post1" --upgrade

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_PREFIX/lib:$LD_LIBRARY_PATH
