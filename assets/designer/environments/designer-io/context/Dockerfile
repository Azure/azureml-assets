FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/designer-io
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

RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade \
        'PyJWT>=2.15.0' 'urllib3>=2.8.0' 'setuptools>=83.0.0' 'msgpack>=1.2.1' && \
    conda run -p $CONDA_PREFIX python -m pip install --no-cache-dir --upgrade \
        'urllib3>=2.8.0' 'setuptools>=83.0.0' 'msgpack>=1.2.1' && \
    find /opt/miniconda $CONDA_PREFIX -path '*/pip/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} + && \
    find /opt/miniconda $CONDA_PREFIX -path '*/pip/_vendor/bom.cdx.json' -type f -delete
