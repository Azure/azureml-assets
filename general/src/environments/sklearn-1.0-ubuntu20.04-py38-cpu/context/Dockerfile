FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.0
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create conda environment
COPY conda_dependencies.yaml conda_dependencies.yaml
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y

# Install pip dependencies
RUN pip install 'matplotlib~=3.5.0' \
                'psutil~=5.8.0' \
                'tqdm~=4.62.0' \
                'pandas~=1.3.0' \
                'scipy~=1.7.0' \
                'numpy~=1.21.0' \
                'ipykernel~=6.0' \
                'azureml-core=={{latest-pypi-version}}' \
                'azureml-defaults=={{latest-pypi-version}}' \
                'azureml-mlflow=={{latest-pypi-version}}' \
                'azureml-telemetry=={{latest-pypi-version}}' \
                'scikit-learn~=1.0.0'
