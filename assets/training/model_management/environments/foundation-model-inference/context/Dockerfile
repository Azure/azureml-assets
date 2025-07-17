FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common wget git runit && \
    add-apt-repository ppa:deadsnakes/ppa -y

# Install Miniconda
ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 -c conda-forge --solver=classic
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /

# Ensure CUDA libraries are linked
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Print CUDA info for debugging (optional)
RUN echo "CUDA VERSION:" && nvcc --version && echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Install dependencies
RUN pip install --no-cache-dir llm-optimized-inference==0.2.31
RUN pip install --no-cache-dir torch==2.7.1

# Optional: FlashAttention from source (after fixing torch/cuda)
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
     cd flash-attention && \
     git checkout v2.7.4.post1 && \
     pip install packaging && \
     pip install --no-build-isolation --no-cache-dir . && \
     cd .. && rm -rf flash-attention

# Cleanup pip cache
RUN rm -rf ~/.cache/pip

# Add service
ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run && chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600

EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
