FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202604.1

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-text-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default && \
    apt-get autoremove -y

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# try updating pip for base and ptca env using conda
RUN conda install pip -n base -y 
RUN conda install pip -n ptca -y

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip \
    numpy~=1.23.5\
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    setuptools \
    wheel=0.46.2 \
    scipy=1.10.1 \
    pybind11=2.10.1 \
    # end conda dependencies
    -c conda-forge -c anaconda

# Ensure additional conda and pip install commands apply to our conda env of interest.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]

# begin pip install
# Install pip dependencies
# Here, we pin sentencepiece since 0.1.98 breaks training. Earlier versions of horovod contain a sev2 vulnerability,
# and earlier versions of tokenizers cause log spam with transformers==4.16.0.
RUN pip install --no-cache-dir \
                # begin pypi dependencies
                azureml-core==1.61.0.post3 \
                azureml-mlflow==1.62.0.post2 \
                azureml-automl-core==1.62.0 \
                azureml-automl-runtime==1.62.0 \
                azureml-defaults==1.62.0 \
                'azure-identity>=1.25.1' \
                'sentencepiece==0.2.1' \
                'filelock>=3.20.3'
                # end pypi dependencies

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client==1.62.0 \
                azureml-train-automl-runtime==1.62.0 \
                azureml-automl-dnn-nlp==1.62.0

# Separate updates for fixing vulnerabilities.
# Doing it  separately from pip install above to avoid conflict with other packages
# We should aim for this list to be empty with new and patched releases
# by fixing dependencies in the base packages
RUN pip list && \
    pip install pyarrow==14.0.2 \
                accelerate==1.12.0 \
                'transformers[sentencepiece,torch]==4.53.0'


# Upgrade starlette, urllib3, bokeh, PyNaCl & filelock
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
RUN pip install --upgrade 'distributed>=2026.1.0' 'cryptography>=46.0.5' 'setuptools>=82.0.1' 'wheel>=0.46.2' 'bokeh>=3.8.2'
# Vulnerability patches for ptca environment
RUN /opt/conda/envs/ptca/bin/pip install --upgrade 'filelock>=3.20.3' 'pillow>=12.1.1' 'cryptography>=46.0.5' 'protobuf>=6.33.5' 'wheel>=0.46.2'
# Fix vendored jaraco.context (GHSA-58pv-8j8x-9vj2) and wheel (GHSA-8rrh-rw8j-w5fx) in ptca/base setuptools
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'
# Base env: fix cryptography, wheel, and setuptools vendored vulns
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
RUN conda run -n base pip install --no-cache-dir --upgrade 'cryptography>=46.0.5' 'wheel>=0.46.2' 'PyJWT>=2.12.0'
RUN /opt/conda/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

RUN /bin/bash -c "source activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
 export CUDACXX=/usr/local/cuda/bin/nvcc && \
 export HOROVOD_BUILD_CUDA_CC_LIST='60,61,70,75,80,86,89,90' && \
 HOROVOD_WITH_PYTORCH=1 \
 HOROVOD_CUDA_HOME=/usr/local/cuda \
 CMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
 pip install --no-cache-dir --no-build-isolation \
 git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d"

RUN rm -rf /opt/conda/pkgs/
# end pip install
