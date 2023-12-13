FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

WORKDIR /

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt update && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.10-dev python3.10-distutils -y
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.10 /usr/bin/python
RUN apt install git -y

# Install pip for Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py' && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Set Python 3.10 as the default python and pip version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

RUN pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121

COPY . .

# mixtral specific
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
RUN pip install git+https://github.com/stanford-futuredata/megablocks.git@5897cd6

RUN pip3 install --no-cache-dir -e .
RUN pip3 list    # show installed packages

# Copy scoring file
COPY src/score.py /var/mlflow_resources/mlflow_score_script.py
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"
ENV AML_APP_ROOT="/var/mlflow_resources"

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

#Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001 8883 8888
CMD [ "runsvdir", "/var/runit" ]
