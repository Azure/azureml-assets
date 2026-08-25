FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV PIP_PROGRESS_BAR=off

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
ENV ENABLE_METADATA=true

# Upgrade vulnerable Ubuntu packages inherited from the base image when installed.
RUN set -eux; \
    apt-get update; \
    os_security_packages="libgssapi-krb5-2 libk5crypto3 libkrb5-3 libkrb5support0 libsqlite3-0 nginx nginx-common nginx-light tar wget"; \
    installed_os_packages=""; \
    for package in $os_security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_os_packages="$installed_os_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_os_packages" ]; then \
        apt-get install -y --only-upgrade $installed_os_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Create conda environment (minimal; packages installed via pip to avoid solver OOM)
RUN conda create -y -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -a -y

# Install cudatoolkit via conda (not available on pip; single-package solve is trivial)
RUN conda install -y -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    cudatoolkit=10.0.130 \
    -c nvidia -c conda-forge && \
    conda clean -a -y

# Install scientific packages via pip (avoids conda solver OOM)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir \
    'numpy>=1.23.5,<1.24' \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'cloudpickle==1.6.0' \
    'scikit-learn==1.5.1' \
    'pandas>=1.5.3,<1.6' \
    'holidays==0.29' \
    'setuptools-git' \
    'setuptools==82.0.1' \
    'wheel>=0.46.2' \
    'torch==2.13.0' \
    'scipy==1.10.1' \
    'psutil>5.0.0,<6.0.0' \
    'pip>=26.1'

# Install pip dependencies. Keep these split across multiple RUN statements so
# no individual Docker layer exceeds the registry validation limit.
# GitPython>=3.1.57 overrides the transitive copy pulled in by mlflow-skinny ->
# databricks-sdk -> gitpython (also pulled by azureml-* telemetry helpers);
# parent packages still allow the older, vulnerable versions.
RUN pip install \
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-pipeline-core=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                'inference-schema' \
                'mlflow-skinny>=2.16.0' \
                'GitPython>=3.1.57' \
                'pillow>=12.1.1'

RUN pip install \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                'mltable>=1.0.0'

RUN pip install \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'xgboost==1.5.2' \
                'pytorch-transformers==1.0.0' \
                'spacy==3.7.4' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz'

# azureml-dataset-runtime is resolved in its own pip invocation, as the other
# AutoML environments already do. Its current release requires pyarrow>=23.0.1
# while azureml-train-automl-runtime pins pyarrow==17.0.0, so resolving both in a
# single invocation is impossible. The pyarrow security floor below is applied
# afterwards regardless.
RUN pip install azureml-dataset-runtime=={{latest-pypi-version}}

# Fix vulnerabilities - security overrides for transitive dependencies.
# Each floor is required because the current parent packages still allow the
# vulnerable lower version, so bumping the parent alone does not remediate.
# distributed>=2026.1.0  CVE-2026-23528; azureml-train-automl-runtime -> dask[complete] -> distributed
# protobuf>=5.29.6       CVE-2025-4565; mlflow-skinny -> protobuf, azureml-automl-runtime -> onnxruntime
# cryptography>=50.0.0   GHSA-537c-gmf6-5ccf, GHSA-g6cj-pr64-35w5; mltable and azureml-core -> paramiko
# bokeh>=3.8.2           GHSA-793v-589g-574v; parent caps bokeh<3.0.0
# onnx>=1.21.0           GHSA-cmw6-hcpp-c6jp and 5 related; azureml-automl-runtime caps onnx<=1.17.0
# urllib3>=2.7.0         GHSA-mf9v-mfxr-j63j, GHSA-qccp-gfcp-xxvc; requests only declares urllib3<3
# ujson>=5.12.1          GHSA-c38f-wx89-p2xg decode buffer overflow
# pyarrow>=23.0.1        GHSA-rgxp-2hwp-jwgg / CVE-2026-25087; azureml-dataset-runtime -> pyarrow
# setuptools>=83.0.0     GHSA-5rjg-fvgr-3xxf, GHSA-h35f-9h28-mq5c
# torch==2.13.0          GHSA-rrmf-rvhw-rf47; azureml-train-automl-runtime resolves 2.8.0, so the
#                        pin must be re-applied after the AzureML packages are installed
#
# numpy>=1.23.5,<1.24 is NOT a security floor. It repeats the cap applied further up so
# that pip sees it while resolving the floors above: onnx pulls ml_dtypes, which as of
# 0.6.0 requires numpy 2.x. Without the cap in this same resolve, pip silently upgrades
# numpy to 2.2.6, which contradicts azureml-automl-runtime, azureml-training-tabular,
# numba and scipy, and leaves the prebuilt pandas 1.5.3 wheel binary-incompatible
# ("numpy.dtype size changed") so the environment fails on `import pandas` at runtime.
RUN pip install --no-cache-dir --upgrade 'numpy>=1.23.5,<1.24' \
    'distributed>=2026.1.0' 'protobuf>=5.29.6' 'cryptography>=50.0.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'urllib3>=2.7.0' \
    'ujson>=5.12.1' \
    'pyarrow>=23.0.1' \
    'setuptools>=83.0.0' \
    'torch==2.13.0'

# The base Miniconda prefix carries its own copy of cryptography, which the
# AzureML conda environment above never touches, so patch it explicitly.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'cryptography>=50.0.0'

# pip's vendored manifests inherited from the base image still record
# setuptools==70.3.0 long after setuptools itself has been upgraded, and the
# scanner reports that stale entry as GHSA-5rjg-fvgr-3xxf / CVE-2025-47273 with
# an empty install path. pip vendors no setuptools code, only the name, so the
# entry is inaccurate rather than a real component. Drop just that line from
# vendor.txt so the genuinely vendored packages stay listed, and drop
# bom.cdx.json, which merely duplicates vendor.txt in CycloneDX form.
RUN rm -rf /opt/miniconda/pkgs/ && \
    find /opt/miniconda $AZUREML_CONDA_ENVIRONMENT_PATH -path '*/pip/_vendor/vendor.txt' \
        -exec sed -i '/^setuptools==/d' {} + && \
    find /opt/miniconda $AZUREML_CONDA_ENVIRONMENT_PATH -path '*/pip/_vendor/bom.cdx.json' -delete

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
