FROM mcr.microsoft.com/mirror/docker/library/ubuntu:22.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:adiscon/v8-stable && \
    apt-get purge -y software-properties-common && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        procps \ 
        libgnutls30 \ 
        libk5crypto3 \ 
        libkrb5-3 \ 
        libkrb5support0 \ 
        libpam-modules \ 
        nginx-light \
        wget \
        runit \
        libtinfo6 \ 
        libncurses6 \ 
        ncurses-bin \ 
        ncurses-base \ 
        libncursesw6 \ 
        libcap2 \ 
        libc6 \    
        libc-bin \
        libtasn1-6 \
        rsyslog \
        build-essential \
        psmisc \
        unzip \
        perl \
        binutils-multiarch \
        binutils \
        libcurl4 \
        libunwind8 \
        systemd \
        libssl3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /usr/share/man/* && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p etc/nginx/sites-available
COPY app etc/nginx/sites-available/app
RUN ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

ADD runit/gunicorn /var/runit/gunicorn
RUN sed -i 's/\r$//g' /var/runit/gunicorn/run
RUN chmod +x /var/runit/gunicorn/run

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
ENV AZUREML_INFERENCE_SERVER_HTTP_ENABLED="True"

EXPOSE 5001

COPY grant_ownership.sh /tmp/

RUN useradd --create-home dockeruser && \
    bash /tmp/grant_ownership.sh && rm -f /tmp/grant_ownership.sh

USER dockeruser

ENV PATH=/opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda update --all -c conda-forge -y && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm -f /tmp/miniconda.sh && \
    find /opt/miniconda -type d -name __pycache__ | xargs rm -rf

WORKDIR /
ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/mlflow
ENV AZUREML_CONDA_DEFAULT_ENVIRONMENT=$AZUREML_CONDA_ENVIRONMENT_PATH

ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

ENV AML_APP_ROOT="/var/mlflow_resources"
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

USER root

COPY mlmonitoring /var/mlflow_resources/mlmonitoring
COPY mlflow_score_script.py /var/mlflow_resources/mlflow_score_script.py
COPY mlflow_hf_score_cpu.py /var/mlflow_resources/mlflow_hf_score_cpu.py
COPY mlflow_hf_score_gpu.py /var/mlflow_resources/mlflow_hf_score_gpu.py

COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y   
USER dockeruser

CMD [ "runsvdir", "/var/runit" ]