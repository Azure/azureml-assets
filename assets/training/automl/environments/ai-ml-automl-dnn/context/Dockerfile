FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH
ENV PIP_PROGRESS_BAR=off

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
ENV ENABLE_METADATA=true

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
    'torch==2.10.0' \
    'scipy==1.10.1' \
    'psutil>5.0.0,<6.0.0' \
    'pip>=26.1'

# Install pip dependencies
# GitPython>=3.1.50 overrides the transitive copy pulled in by mlflow-skinny ->
# databricks-sdk -> gitpython (also pulled by azureml-* telemetry helpers).
# Required for GHSA-7545-fcxq-7j24, GHSA-mv93-w799-cj2w, GHSA-v87r-6q3f-2j67;
# parent packages still allow the older versions, so this direct floor is needed.
RUN pip install \
                azureml-core=={{latest-pypi-version}} \
                azureml-mlflow=={{latest-pypi-version}} \
                azureml-pipeline-core=={{latest-pypi-version}} \
                azureml-telemetry=={{latest-pypi-version}} \
                azureml-interpret=={{latest-pypi-version}} \
                azureml-responsibleai=={{latest-pypi-version}} \
                azureml-automl-core=={{latest-pypi-version}} \
                azureml-automl-runtime=={{latest-pypi-version}} \
                azureml-train-automl-client=={{latest-pypi-version}} \
                azureml-train-automl-runtime=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                'inference-schema' \
                'mlflow-skinny>=2.16.0' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'xgboost==1.5.2' \
                'mltable>=1.0.0' \
                'pytorch-transformers==1.0.0' \
                'GitPython>=3.1.50' \
                'spacy==3.7.4' \
                'pillow>=12.1.1' \
                'https://aka.ms/automl-resources/packages/en_core_web_sm-3.7.1.tar.gz'

# Fix vulnerabilities - security overrides for transitive dependencies.
# Parent research refreshed 2026-05-26; direct floors are kept only where the
# current parent packages still allow vulnerable lower versions.
#
# distributed>=2026.1.0    CVE-2026-23528  XSS-to-RCE via Dask dashboard proxy
#                          Chain: azureml-train-automl-runtime 1.62.0 ->
#                          dask[complete]<=2023.2.0 -> distributed. Latest dask
#                          only aligns distributed to the dask release; it does
#                          not force this security floor.
#
# protobuf>=5.29.6         CVE-2025-4565   DoS via recursive protobuf messages (pure-Python)
#                          Chain (L1): mlflow-skinny -> protobuf. Latest
#                          mlflow-skinny 3.12.0 still allows protobuf>=3.12.0.
#                          Chain (L2): azureml-automl-runtime -> onnxruntime -> protobuf
#
# cryptography>=46.0.5     CVE-2026-26007  EC subgroup validation bypass (ECDH key leak)
#                          Chain (L1): mltable -> cryptography
#                          Chain (L2): azureml-core -> paramiko -> cryptography.
#                          Latest mltable 1.6.3 and paramiko 3.5.1 keep loose
#                          cryptography floors.
#
# bokeh>=3.8.2             GHSA-793v-589g-574v  conda env installs 2.4.3, pip can't auto-upgrade
#                          Chain (L1): azureml-train-automl-runtime 1.62.0 ->
#                          bokeh<3.0.0 and dask[complete]<=2023.2.0.
#
# onnx>=1.21.0             GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6,
#                          GHSA-p433-9wv8-28xj, GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m
#                          azureml-automl-runtime pins onnx<=1.17.0; latest release (1.62.0)
#                          still carries that upper bound, so a parent-package upgrade cannot
#                          resolve the CVEs.  Override required.
#                          Chain (L1): azureml-automl-runtime -> onnx
#
# urllib3>=2.7.0           GHSA-mf9v-mfxr-j63j, GHSA-qccp-gfcp-xxvc
#                          Chain (L1): requests -> urllib3  (pulled by ~every azureml-* pkg,
#                          mlflow-skinny, mltable, azure-* SDKs).  Latest `requests` 2.32.5
#                          still declares `urllib3<3`, never `>=2.7`, so upgrading any
#                          intermediate parent cannot raise the floor.  Direct override required.
#
# ujson>=5.12.1            GHSA-c38f-wx89-p2xg  (decode buffer overflow)
#                          Current parent metadata does not declare a tight ujson
#                          floor, so the floor can only be raised via a direct
#                          override here.
# pyarrow>=23.0.1          GHSA-rgxp-2hwp-jwgg / CVE-2026-25087 use-after-free
#                          in Apache Arrow C++. Chain (L1):
#                          azureml-dataset-runtime -> pyarrow.
# torch==2.10.0            GHSA-vgrw-7cvw-pwgx, GHSA-qfhq-4f3w-5fph.
#                          Direct AutoML DNN dependency; the later AzureML /
#                          pytorch-transformers install can resolve back to
#                          torch 2.8.0, so enforce the patched final version.
RUN pip install --no-cache-dir --upgrade 'distributed>=2026.1.0' 'protobuf>=5.29.6' 'cryptography>=46.0.5' \
    'bokeh>=3.8.2' \
    'onnx>=1.21.0' \
    'torch==2.10.0' \
    'urllib3>=2.7.0' \
    'ujson>=5.12.1' \
    'pyarrow>=23.0.1'
RUN rm -rf /opt/miniconda/pkgs/

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
