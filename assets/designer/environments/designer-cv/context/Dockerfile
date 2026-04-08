FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}
# Apply Ubuntu security patches for high/medium vulnerabilities tracked in Designer images.
RUN apt-get update && apt-get install --no-install-recommends -y --only-upgrade \
    curl \
    libcurl3t64-gnutls \
    libcurl4t64 \
    libssh-4 \
    libpython3.12-minimal \
    libpython3.12-stdlib \
    python3.12 \
    python3.12-minimal && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/designer-cv
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

