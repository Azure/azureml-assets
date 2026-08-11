FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260810.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# distributed/bokeh/onnx/msgpack: parent packages cap these below patched versions.
# GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
# GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (onnx)
# cryptography is a pinned transitive dep of azureml-core; fixes GHSA-537c-gmf6-5ccf.
# pyarrow is a pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# torch is a pinned transitive dep of azureml-contrib-automl-dnn-forecasting; fixes GHSA-rrmf-rvhw-rf47.
# msgpack is a pinned transitive dep of distributed; fixes GHSA-6v7p-g79w-8964.
COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python -m pip install --upgrade --no-cache-dir 'pip>=26.1.2' && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=48.0.1' 'GitPython>=3.1.55' 'setuptools>=83.0.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pyarrow==23.0.1' \
    'torch==2.13.0' \
    'msgpack>=1.2.1' && \
    rm -rf $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip-26.1.1.dist-info && \
    rm -f $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json \
        $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip/_vendor/vendor.txt && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true

RUN apt-get update && \
    apt-get install --only-upgrade -y --no-install-recommends \
        tar=1.34+dfsg-1ubuntu0.1.22.04.6 \
        libsqlite3-0=3.37.2-2ubuntu0.7 \
        nginx-light=1.18.0-6ubuntu14.18 \
        libnginx-mod-http-echo=1.18.0-6ubuntu14.18 \
        nginx-common=1.18.0-6ubuntu14.18 \
        libnginx-mod-http-geoip2=1.18.0-6ubuntu14.18 \
        wget=1.21.2-2ubuntu1.4 \
        libkrb5support0=1.19.2-2ubuntu0.8 \
        libgssapi-krb5-2=1.19.2-2ubuntu0.8 \
        libk5crypto3=1.19.2-2ubuntu0.8 \
        libkrb5-3=1.19.2-2ubuntu0.8 \
        libpam-runtime=1.4.0-11ubuntu2.7 \
        libpam0g=1.4.0-11ubuntu2.7 \
        libpam-modules=1.4.0-11ubuntu2.7 \
        libpam-modules-bin=1.4.0-11ubuntu2.7 \
        rsyslog=8.2112.0-2ubuntu2.4 \
        libc-dev-bin=2.35-0ubuntu3.14 \
        libc6-dev=2.35-0ubuntu3.14 \
        libc6=2.35-0ubuntu3.14 \
        libc-bin=2.35-0ubuntu3.14 && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib