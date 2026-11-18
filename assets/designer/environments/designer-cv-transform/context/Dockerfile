FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/designer-cv-transform
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

# Reassert November 2026 security floors after environment creation.
RUN set -eux; \
    apt-get update; \
    security_packages="curl libcurl3t64-gnutls libcurl4t64 libbrotli1 libxml2 openssh-sftp-server openssh-client openssh-server zlib1g nginx nginx-common nginx-light"; \
    installed_packages=""; \
    for package in $security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_packages="$installed_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_packages" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade $installed_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade \
        'certifi>=2026.10.19' 'setuptools>=84.1.0' && \
    "$CONDA_PREFIX/bin/python" -m pip install --no-cache-dir --force-reinstall --no-deps \
        'setuptools>=84.1.0' && \
    find /opt/miniconda "$CONDA_PREFIX" -path '*/setuptools-84.0.0*.egg-info' -exec rm -rf {} + && \
    rm -rf /opt/miniconda/pkgs/

RUN find /opt/miniconda $CONDA_PREFIX -path '*/pip/_vendor/bom.cdx.json' -delete
