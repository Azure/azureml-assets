FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install git -y

ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget runit
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/default

# Create conda environment with py310
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge --solver=classic

ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH

ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /

# When copied to assets repo, change to install from public pypi
RUN pip install llm-optimized-inference==0.2.31 --no-cache-dir

# torch installation
RUN pip install --no-cache-dir torch==2.7.1

# Accept a build-time argument (default is FALSE = not in CI)
ARG SKIP_FLASH_ATTN_BUILD=FALSE
ENV SKIP_FLASH_ATTN_BUILD=${SKIP_FLASH_ATTN_BUILD}

# Clone flash-attention repo (so the source is still available)
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.7.4.post1 && \
    cd ..

# Conditionally install flash-attn only if SKIP_FLASH_ATTN_BUILD is not TRUE
RUN if [ "$SKIP_FLASH_ATTN_BUILD" != "TRUE" ]; then \
        cd flash-attention && \
        pip install packaging && \
        pip install --no-build-isolation --no-cache-dir . && \
        cd .. && rm -rf flash-attention ; \
    else \
        echo "Skipping flash-attention install (CI mode)" ; \
    fi

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run
RUN chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
