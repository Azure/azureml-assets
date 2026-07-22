FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-text-gpu
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

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

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip \
    numpy~=1.23.5\
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    setuptools=82.0.1 \
    wheel=0.46.3 \
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
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                'azure-identity>=1.25.1' \
                'sentencepiece==0.2.1' \
                'filelock>=3.20.3'
                # end pypi dependencies

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-automl-dnn-nlp=={{latest-pypi-version}}

# Separate updates for fixing vulnerabilities.
# Doing it  separately from pip install above to avoid conflict with other packages
# We should aim for this list to be empty with new and patched releases
# by fixing dependencies in the base packages
RUN pip install pyarrow==14.0.2 \
                accelerate==1.12.0 

# Override transformers to fix GHSA-69w3-r845-3855
# Root cause: azureml-automl-dnn-nlp==1.62.0 (latest as of 2026-05-26)
# pins transformers==4.53.0; cannot upgrade parent.
RUN pip install --no-cache-dir --no-deps 'transformers[sentencepiece,torch]==5.5.4'

# Security: upgrade pip to fix self-update vulnerabilities in the AzureML conda env.
# pip is its own parent, so explicit upgrade is the only remediation.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1.2'

# Upgrade bokeh, cryptography, and onnx in the AZUREML conda env (py3.10).
# NOTE: AzureML packages do not force the fixed cryptography floor; use >=48.0.1.
# Override onnx to fix GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6, GHSA-p433-9wv8-28xj, GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m, and GHSA-x4hj-rm42-cggg
# Root cause: azureml-automl-runtime==1.62.0 (latest) pins onnx<=1.17.0; cannot upgrade parent
RUN pip install --upgrade 'distributed>=2026.1.0' 'cryptography>=48.0.1' 'bokeh>=3.8.2' 'onnx>=1.22.0'

# Security: upgrade aiohttp to fix GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8,
# and follow-up aiohttp advisories requiring >=3.14.1 in the AzureML conda env.
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.1'

# torch is pinned by the ACPT base image in the ptca env and by
# azureml-automl-dnn-nlp in this AzureML env; upgrade both for GHSA-vgrw-7cvw-pwgx
# and GHSA-qfhq-4f3w-5fph while keeping CUDA 12.6 wheels.
RUN pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 'torch==2.10.0+cu126'
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 'torch==2.10.0+cu126'

# Apply the same setuptools vendored-dependency fix in the AzureML conda env.
RUN pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

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
