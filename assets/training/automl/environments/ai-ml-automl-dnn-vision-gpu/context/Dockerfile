FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
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

# Update pip in base and ptca environments
RUN conda install pip -n base -y
RUN conda install pip -n ptca -y

# Create conda environment
# begin conda create
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip \
    numpy~=1.23.5\
    libffi=3.4.4 \
    pycocotools=2.0.4 \
    shap=0.39.0 \
    llvmlite=0.39.1 \
    scipy=1.10.1 \
    setuptools=82.0.1 \
    wheel=0.46.2 \
    tbb=2021.1.1 \
    # end conda dependencies
    -c conda-forge -c cerebis && \
    conda clean -a -y
# end conda create

# begin pip install

# Install pip dependencies
RUN pip install --no-cache-dir \
                 # begin pypi dependencies
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                'azureml-dataprep>=2.24.4' \
                'azure-identity>=1.25.1' \
                'pillow==12.2.0'
                # end pypi dependencies

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-automl-dnn-vision=={{latest-pypi-version}}
                
# Vulnerability patches for conda environment
# NOTE: azureml-mlflow~=1.62.0 pins cryptography<46.0.0; upgrading anyway for CVE fix
# NOTE: azureml-automl-runtime pins onnx<=1.17.0,>=1.16.1; force-installing onnx>=1.21.0 to fix
#       GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6, GHSA-p433-9wv8-28xj,
#       GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m. Parent azureml-automl-runtime (1.62.0) cannot
#       be upgraded as latest still constrains onnx<=1.17.0.
RUN pip install --no-cache-dir --upgrade \
                'cryptography>=46.0.5' \
                'distributed>=2026.1.0' \
                'mlflow-skinny>=2.16.0' \ 
                'bokeh>=3.8.2' \
                'pillow==12.2.0' \
                'onnx>=1.21.0'




# Vulnerability patches for ptca environment
RUN /opt/conda/envs/ptca/bin/pip install --upgrade \
                'pillow==12.2.0' 'filelock>=3.20.3' 'cryptography>=46.0.5' 'protobuf>=6.33.5' 'wheel>=0.46.2' \
                'pytest>=9.0.3'
# setuptools resolver picks wrong version due to dep conflicts; force install to fix jaraco.context vuln (GHSA-58pv-8j8x-9vj2)
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

RUN conda run -n base pip install --no-cache-dir --upgrade \
                'cryptography>=46.0.5' \
                'wheel>=0.46.2' \
                'PyJWT>=2.12.0' \
                'aiohttp>=3.13.4'
# PyJWT 2.10.1 (CVE-2026-32597) is installed in the base conda env (python3.13) from ACPT base image; manually upgrading since base image hasn't been patched yet
# Fix vendored jaraco.context (GHSA-58pv-8j8x-9vj2) and wheel (GHSA-8rrh-rw8j-w5fx) in base setuptools
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN /opt/conda/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

RUN rm -rf /opt/conda/pkgs/

# end pip install
ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
