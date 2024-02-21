FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

RUN apt install git -y

ENV MINICONDA_VERSION py310_23.10.0-1
ENV PATH /opt/miniconda/bin:$PATH
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
    pip<=24.0 -c conda-forge 

ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH
ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

WORKDIR /

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Needed for megablocks
RUN pip install torch~=2.1.2 --index-url https://download.pytorch.org/whl/cu121

COPY . .

# mixtral specific
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
RUN pip install git+https://github.com/stanford-futuredata/megablocks.git@5897cd6

RUN pip install --no-cache-dir -e .
RUN pip list    # show installed packages

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

ADD runit_folder/api_server /var/runit/api_server
RUN sed -i 's/\r$//g' /var/runit/api_server/run
RUN chmod +x /var/runit/api_server/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001
CMD [ "runsvdir", "/var/runit" ]
