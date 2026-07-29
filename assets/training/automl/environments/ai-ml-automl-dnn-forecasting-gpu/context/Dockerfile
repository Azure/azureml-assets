FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-forecasting-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true

# Upgrade vulnerable OS packages installed by the Ubuntu 22.04 parent image.
# Includes OpenSSL (USN-8414-1), systemd (USN-8402-1), nginx
# (USN-8354-1/USN-8398-1), XZ Utils (USN-8362-1), and earlier base-image USNs.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade; \
    for package in \
        curl \
        libcurl3-gnutls \
        libcurl4 \
        liblzma5 \
        libnghttp2-14 \
        libnginx-mod-http-echo \
        libnginx-mod-http-geoip2 \
        libssl3 \
        libsystemd0 \
        libudev1 \
        nginx-common \
        nginx-light \
        openssh-client \
        openssh-server \
        openssh-sftp-server \
        openssl \
        sed \
        xz-utils; \
    do \
        if dpkg-query -W -f='${db:Status-Abbrev}' "$package" 2>/dev/null | grep -q '^ii '; then \
            DEBIAN_FRONTEND=noninteractive apt-get install --only-upgrade -y --no-install-recommends "$package"; \
        fi; \
    done; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
        libboost-system-dev \
        libboost-filesystem-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip in the base miniconda (/opt/miniconda) to fix CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9.
# The base miniconda ships with the parent image (mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04)
# and is independent of the conda env created below at $AZUREML_CONDA_ENVIRONMENT_PATH.
# Pinning 'pip>=26.1' inside the env (line below) does not patch /opt/miniconda's pip, which is
# still flagged by SCA scanners. No parent-package upgrade is possible because pip is the package
# itself; the only fix is upgrading pip directly to >= 26.1.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'pip>=26.1'

# Create conda environment (minimal — packages installed via pip to avoid solver OOM)
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -a -y

# Install scientific packages + pip/setuptools/wheel via pip (avoids conda solver OOM)
# Pin setuptools below 82 so pkg_resources remains available for AzureML packages.
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
    'numpy>=1.23.5,<1.24' \
    'scikit-learn==1.5.1' \
    'pandas>=1.5.3,<1.6' \
    'scipy==1.10.1' \
    'psutil>=5.2.2,<6.0.0' \
    'pip>=26.1' \
    'setuptools==81.0.0' \
    'wheel>=0.46.2'

# Install pip dependencies
# GitPython>=3.1.41 is required for https://github.com/advisories/GHSA-2mqj-m65w-jghx and is not available in conda
RUN pip install --no-cache-dir \
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                'azure-identity>=1.25.1' \
                'inference-schema' \
                'xgboost==3.2.0' \
                'GitPython>=3.1.41' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0' 

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-contrib-automl-dnn-forecasting=={{latest-pypi-version}}

# Fix vulnerabilities - overrides needed because parent packages cap vulnerable versions
# distributed>=2026.1.0: CVE-2026-23528 (via azureml-train-automl-runtime -> dask[complete])
# cryptography>=46.0.5: CVE-2026-26007 (via azureml-mlflow, azure-identity, azureml-core -> msal/pyopenssl)
# mlflow-skinny>=2.16.0: security fixes (via azureml-mlflow, azureml-contrib-automl-dnn-forecasting)
# protobuf>=5.29.6: CVE-2026-0994 (via mlflow-skinny, azureml-automl-runtime -> onnx/onnxruntime)
# pillow>=12.1.1: CVE-2026-25990 (via matplotlib, bokeh, tensorboard)
# bokeh>=3.8.2: GHSA-793v-589g-574v CSWSH (overrides azureml-train-automl-runtime's bokeh<3.0.0 cap)
# onnx>=1.21.0: GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6, GHSA-538c-55jv-c5g9,
#   GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (via azureml-automl-runtime -> onnxconverter-common/skl2onnx)
# pyarrow>=23.0.1: GHSA-rgxp-2hwp-jwgg / CVE-2026-25087 (via azureml-dataset-runtime)
RUN pip install --no-cache-dir --upgrade 'distributed>=2026.1.0' 'cryptography>=46.0.5' 'mlflow-skinny>=2.16.0' \
                          'protobuf>=5.29.6' 'pillow>=12.1.1' \
                          'bokeh>=3.8.2' \
                          'onnx>=1.21.0' \
                          'pyarrow>=23.0.1'  # pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg / CVE-2026-25087
RUN pip install --no-cache-dir torch==2.8.0

RUN /bin/bash -c "source activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
 export CUDACXX=/usr/local/cuda/bin/nvcc && \
 export HOROVOD_BUILD_CUDA_CC_LIST='60,61,70,75,80,86,89,90' && \
 HOROVOD_WITHOUT_TENSORFLOW=1 \
 HOROVOD_WITH_PYTORCH=1 \
 HOROVOD_CUDA_HOME=/usr/local/cuda \
 CMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
 pip install --no-cache-dir --no-build-isolation \
 git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d"

RUN rm -rf /opt/miniconda/pkgs/
