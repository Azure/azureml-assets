FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

USER root

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
# ENV AML_APP_ROOT="/var/mlflow_resources"
# ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"

ENV ENABLE_METADATA=true
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Defensive OS-package upgrade: pulls in any USN-patched packages published after
# the base image was built. The latest parent image
# (mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04, resolved via
# {{latest-image-tag}}) already ships the fixes for USN-8222-1 (openssh),
# USN-8227-1 (curl/libcurl*), USN-8229-1 (sed) and USN-8233-1 (libnghttp2-14).
# Packages listed below are explicitly reinstalled to force pickup of patched
# versions even when an older base layer is cached (same pattern used across
# other automl Dockerfiles).
# USN-8284-1: libgnutls30 3.7.3-4ubuntu1.8 -> 3.7.3-4ubuntu1.9
# USN-8292-1: libarchive13 3.6.0-1ubuntu1.6 -> 3.6.0-1ubuntu1.7
# USN-8319-1: libgcrypt20 1.9.4-3ubuntu3 -> 1.9.4-3ubuntu3.2
# USN-8354-1: nginx-common/libnginx-mod-*/nginx-light 1.18.0-6ubuntu14.11 -> 1.18.0-6ubuntu14.12
# USN-8362-1: liblzma5/xz-utils 5.2.5-2ubuntu1 -> 5.2.5-2ubuntu1.1
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install --reinstall -y \
        libgnutls30 \
        libarchive13 \
        libgcrypt20 \
        liblzma5 xz-utils && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade \
        nginx-common \
        libnginx-mod-http-geoip2 \
        nginx-light \
        libnginx-mod-http-echo && \
    apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip remediated to >=26.1 (GHSA-jp4c-xjxw-mgf9 / CVE-2026-6357).
# Why a two-step install instead of a single `conda create ... pip=26.1`:
#   The Anaconda defaults channel currently ships only pip 26.0.1; pip 26.1 lives
#   on conda-forge (verified via repodata: defaults has [25.1, 25.2, 25.3, 26.0.1],
#   conda-forge has 26.1.1). Mixing channels in the conda create solve has caused
#   solver instability for this image in the past, so we keep `conda create` on
#   defaults and explicitly upgrade pip afterwards. This mirrors the same pattern
#   used in assets/training/automl/environments/ai-ml-automl/context/Dockerfile.
# `python -m pip` is the recommended self-upgrade form (avoids running the pip
# script while it is being replaced). Upper bound `<27` guards against an
# unattended jump to a future major release that could break the rest of the
# image. opt/miniconda's pip is already 26.1.1 in the latest parent base image,
# so no secondary upgrade for that prefix is required here.
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10 pip=26.0 -y && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python -m pip install --no-cache-dir --upgrade 'pip>=26.1,<27' && \
    conda clean -a -y

# Security: upgrade idna in the base miniconda env (GHSA-65pc-fj4g-8rjx).
# VCM SCA scans /opt/miniconda/lib/python3.12/site-packages/ and detects idna<3.15.
# The base image ships idna 3.13; 3.15 fixes the ReDoS via crafted hostname labels.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'idna>=3.15'

###############################
# Pre-Build LightGBM
###############################
RUN pip install --upgrade lightgbm==4.6.0

###############################
# Install GPU LightGBM and XgBoost
###############################
RUN pip install --upgrade --force-reinstall xgboost==1.5.2 pandas==1.5.3

# begin conda create
# Install ncurses via conda (C library, not available on pip; single-package solve is trivial)
RUN conda install -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    ncurses=6.5 \
    -c conda-forge -y

# Install packages via pip (avoids conda solver OOM)
RUN pip install --no-cache-dir \
    'numpy>=1.23.5,<1.24' \
    'scikit-learn==1.5.1' \
    'holidays==0.29' \
    'setuptools-git' \
    'wheel>=0.46.2' \
    'scipy==1.10.1' \
    'joblib==1.2.0'

# Ensure future conda and pip calls are tied to this conda environment,
# and we don't just get lucky from the implicit PATH ordering.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]
# end conda create

# begin pip install
# Install pip dependencies
RUN pip install \
                # begin pypi dependencies
                azureml-core=={{latest-pypi-version}} \
                azureml-pipeline-core=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                'azureml-model-management-sdk==1.0.1b6.post1' \
                'mlflow-skinny==2.15.1' \
                'azure-identity>=1.25.1' --no-cache-dir

RUN pip install \
                'inference-schema' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'pytorch-transformers==1.0.0' --no-cache-dir

RUN pip install \
                'spacy==3.7.4' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0' \
                'jinja2>=3.1.6' --no-cache-dir
                # end pypi dependencies

# Fix vulnerabilities - overrides needed because parent packages cap vulnerable versions
# torchvision==0.23.0, torch==2.8.0: CVE-2025-32434 RCE via torch.load (via pytorch-transformers -> torch, horovod)
# distributed>=2026.1.0: CVE-2026-23528 (via azureml-train-automl-runtime -> dask[complete] -> distributed)
# mlflow-skinny>=2.16.0: security fixes (via azureml-mlflow -> mlflow-skinny, also installed directly)
# protobuf>=5.29.6: CVE-2026-0994 DoS recursion (via mlflow-skinny -> protobuf, azureml-automl-runtime -> onnx/onnxruntime)
# pillow>=12.1.1: CVE-2026-25990 heap buffer overflow (via torchvision -> pillow, prophet -> matplotlib -> pillow)
# bokeh>=3.8.2: GHSA-793v-589g-574v CSWSH (overrides azureml-train-automl-runtime's bokeh<3.0.0 cap)
# onnx>=1.21.0: GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6, GHSA-538c-55jv-c5g9,
#   GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (overrides azureml-automl-runtime's onnx<=1.17.0 cap)
RUN pip install --upgrade torchvision==0.23.0 torch==2.8.0 --no-cache-dir
RUN pip install --upgrade 'distributed>=2026.1.0' 'mlflow-skinny>=2.16.0' \
                          'protobuf>=5.29.6' 'pillow>=12.1.1' \
                          'bokeh>=3.8.2' \
                          'onnx>=1.21.0' \
                          'starlette>=1.0.1'  # GHSA-86qp-5c8j-p5mr: DoS via malformed multipart (0.52.1 -> >=1.0.1)

# --- End Vulnerability Fixes ---


RUN /bin/bash -c "source activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
 export CUDACXX=/usr/local/cuda/bin/nvcc && \
 export HOROVOD_BUILD_CUDA_CC_LIST='60,61,70,75,80,86,89,90' && \
 HOROVOD_WITH_PYTORCH=1 \
 HOROVOD_CUDA_HOME=/usr/local/cuda \
 CMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
 pip install --no-cache-dir --no-build-isolation \
 git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d"

RUN rm -rf /opt/miniconda/pkgs/

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
