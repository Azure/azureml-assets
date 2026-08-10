FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/automl
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

ENV ENABLE_METADATA=true
ENV AUTOML_ENABLE_SAFE_FORECASTING_MODEL_EXPORT=true

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
        libkrb5-3=1.19.2-2ubuntu0.8 && \
    rm -rf /var/lib/apt/lists/*

COPY conda_dependencies.yaml .
RUN conda env create -p $AZUREML_CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python -m pip install --upgrade --no-cache-dir 'pip>=26.1.2' && \
    rm -rf $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages/pip-26.1.1.dist-info && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

# distributed/bokeh/onnx: parent packages cap these below patched versions.
# GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
# GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m (onnx)
# cryptography is a pinned transitive dep of azureml-core; fixes GHSA-537c-gmf6-5ccf.
# pyarrow is a pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# torch is a pinned transitive dep of azureml-contrib-automl-dnn-forecasting; fixes GHSA-rrmf-rvhw-rf47.
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --upgrade --no-cache-dir \
    'cryptography>=48.0.1' 'setuptools>=79.0.0' 'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'pyarrow==23.0.1' \
    'torch==2.13.0'

# azureml-automl-runtime requires numpy<=1.23.5, while skops 0.14 declares
# numpy>=1.25. The skops.io paths used by this environment are validated against
# numpy 1.23.5 below, so install skops without changing the supported AutoML stack.
COPY skops_compatibility_test.py .
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --no-deps \
    'skops==0.14.0' && \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH python skops_compatibility_test.py && \
    rm skops_compatibility_test.py

RUN rm -rf /opt/miniconda/pkgs/

# Avoid ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AZUREML_CONDA_ENVIRONMENT_PATH/lib