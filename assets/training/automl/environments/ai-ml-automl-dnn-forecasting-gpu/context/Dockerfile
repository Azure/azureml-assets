FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-forecasting-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true

# Pick up patched Ubuntu packages when the parent image or apt-installed
# dependencies contain vulnerable versions.
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

RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade 'pip>=26.1' && \
    find /opt/miniconda -path '*/pip/_vendor/bom.cdx.json' -type f -delete && \
    find /opt/miniconda -path '*/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} +

# Create conda environment; packages are installed via pip to avoid solver OOM.
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -a -y

# Install scientific packages plus packaging tools via pip.
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
    'numpy>=1.23.5,<1.24' \
    'scikit-learn==1.5.1' \
    'pandas>=1.5.3,<1.6' \
    'scipy==1.10.1' \
    'psutil>=5.2.2,<6.0.0' \
    'pip>=26.1' \
    'setuptools>=83.0.0' \
    'wheel>=0.46.2' && \
    find "$AZUREML_CONDA_ENVIRONMENT_PATH" -path '*/pip/_vendor/bom.cdx.json' -type f -delete && \
    find "$AZUREML_CONDA_ENVIRONMENT_PATH" -path '*/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} +

# Install AzureML dependencies with security constraints for transitive packages
# pulled in by the AzureML AutoML stack.
RUN printf '%s\n' \
                '# GitPython pinned transitive dep of azureml-core; fixes GHSA-2f96-g7mh-g2hx, GHSA-956x-8gvw-wg5v, GHSA-v396-v7q4-x2qj, GHSA-94p4-4cq8-9g67, GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, and GHSA-r9mr-m37c-5fr3' \
                'GitPython>=3.1.55' \
                '# msgpack pinned transitive dep of AzureML AutoML/Dask; fixes GHSA-6v7p-g79w-8964' \
                'msgpack>=1.2.1' \
                '# numpy is NOT a security floor. onnx pulls ml_dtypes, which as of 0.6.0 requires numpy>=2.0.0; without this cap the security-override resolve below silently upgrades numpy to 2.x, contradicting azureml-automl-runtime/numba/scipy and leaving the prebuilt pandas 1.5.3 wheel binary-incompatible, so the image fails on import pandas with "numpy.dtype size changed".' \
                'numpy>=1.23.5,<1.24' \
                '# pillow pinned transitive dep of matplotlib/bokeh/tensorboard; fixes GHSA-8v84-f9pq-wr9x, GHSA-9hw9-ch79-4vh6, GHSA-vjc4-5qp5-m44j, GHSA-pg7v-jwj7-p798, GHSA-45hq-cxwh-f6vc, GHSA-phj9-mv4w-65pm, GHSA-62p4-gmf7-7g93, and GHSA-5x94-69rx-g8h2' \
                'pillow>=12.3.0' \
                '# pydantic-settings pinned transitive dep of AzureML/MLflow; fixes GHSA-4xgf-cpjx-pc3j' \
                'pydantic-settings>=2.14.2' \
                '# pyasn1 pinned transitive dep of azure-identity/msal; fixes GHSA-hm4w-wwcw-mr6r and GHSA-8ppf-4f7h-5ppj' \
                'pyasn1>=0.6.4' \
                '# setuptools pinned packaging tool in this conda env; fixes GHSA-h35f-9h28-mq5c' \
                'setuptools>=83.0.0' \
                '# sqlparse pinned transitive dep of mlflow-skinny; fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f' \
                'sqlparse>=0.6.0' \
                > /tmp/security-constraints.txt && \
    pip install --no-cache-dir -c /tmp/security-constraints.txt \
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                'inference-schema' \
                'xgboost==3.2.0' \
                'GitPython>=3.1.55' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz' \
                'py-cpuinfo==5.0.0' 

RUN pip install --no-cache-dir --upgrade --no-deps 'azure-identity>=1.25.1'

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir -c /tmp/security-constraints.txt \
                azureml-train-automl=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-contrib-automl-dnn-forecasting=={{latest-pypi-version}}

# Security overrides needed because parent packages cap vulnerable versions.
# distributed>=2026.1.0: CVE-2026-23528 (via azureml-train-automl-runtime -> dask[complete])
# cryptography>=46.0.5: CVE-2026-26007 (via azureml-mlflow, azure-identity, azureml-core -> msal/pyopenssl)
# mlflow-skinny>=2.16.0: security fixes (via azureml-mlflow, azureml-contrib-automl-dnn-forecasting)
# protobuf>=5.29.6: CVE-2026-0994 (via mlflow-skinny, azureml-automl-runtime -> onnx/onnxruntime)
# pillow>=12.3.0: current Pillow GHSA findings (via matplotlib, bokeh, tensorboard)
# bokeh>=3.8.2: GHSA-793v-589g-574v CSWSH (overrides azureml-train-automl-runtime's bokeh<3.0.0 cap)
# onnx>=1.21.0: GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6, GHSA-538c-55jv-c5g9,
#   GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (via azureml-automl-runtime -> onnxconverter-common/skl2onnx)
# pyarrow>=23.0.1: GHSA-rgxp-2hwp-jwgg / CVE-2026-25087 (via azureml-dataset-runtime)
# sqlparse>=0.6.0: GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, GHSA-pwgv-4x5q-6m9f (via mlflow-skinny)
RUN pip install --no-cache-dir --upgrade -c /tmp/security-constraints.txt \
                          'distributed>=2026.1.0' \
                          'cryptography>=46.0.5' \
                          'mlflow-skinny>=2.16.0' \
                          'protobuf>=5.29.6' \
                          'pillow>=12.3.0' \
                          'bokeh>=3.8.2' \
                          'onnx>=1.21.0' \
                          'pyarrow>=23.0.1' \
                          'sqlparse>=0.6.0'

# torch pinned for azureml-contrib-automl-dnn-forecasting; fixes GHSA-rrmf-rvhw-rf47,
# GHSA-vgrw-7cvw-pwgx, and GHSA-qfhq-4f3w-5fph.
RUN pip install --no-cache-dir torch==2.13.0 && \
    rm -f /tmp/security-constraints.txt

RUN /bin/bash -c "source activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
 export CUDACXX=/usr/local/cuda/bin/nvcc && \
 export HOROVOD_BUILD_CUDA_CC_LIST='60,61,70,75,80,86,89,90' && \
 HOROVOD_WITHOUT_TENSORFLOW=1 \
 HOROVOD_WITH_PYTORCH=1 \
 HOROVOD_CUDA_HOME=/usr/local/cuda \
 CMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
 pip install --no-cache-dir --no-build-isolation \
 git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d"

RUN set -eux; \
    find / -xdev -path '*/site-packages/pip/_vendor/bom.cdx.json' -type f -delete; \
    find /opt /azureml-envs -path '*/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} +; \
    rm -rf /opt/miniconda/pkgs/
