FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:{{latest-image-tag}}
WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-ex
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Create conda environment
COPY conda_dependencies.yaml .
RUN sed -i 's#{{latest-pypi-version}}#1.48.0#g' conda_dependencies.yaml
RUN sed -i 's#{{latest-sklreanex-version}}#2023.0.0#g' conda_dependencies.yaml
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
