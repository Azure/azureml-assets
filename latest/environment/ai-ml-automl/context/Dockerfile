FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260715.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
ENV ENABLE_METADATA=true
ENV PYTHONIOENCODING=utf-8
ENV PIP_NO_COLOR=1
ENV PIP_PROGRESS_BAR=off

# Create conda environment (minimal -- packages installed via pip to avoid solver OOM)
RUN conda create -q -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -q -a -y

# Security overrides for packages installed in independent Python prefixes:
# - pip>=26.1.2 fixes GHSA-wf93-45jw-7689. pip is its own parent,
#   so explicit upgrades are the only available remediation.
# - PyJWT>=2.13.0 is pinned as a transitive dep brought in by the base image
#   (mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04), fixing GHSA-993g-76c3-p5m4
#   and GHSA-jq35-7prp-9v3f.
# - idna>=3.15 is pinned as a transitive dep brought in by the base image,
#   fixing GHSA-65pc-fj4g-8rjx.
# - py-rattler>=0.24.0 is pinned as a transitive dep brought in by the base image,
#   fixing GHSA-q53q-5r4j-5729.
# - cryptography>=48.0.1 is pinned as a transitive dep brought in by the base image
#   and its ptca conda env, fixing GHSA-537c-gmf6-5ccf.
RUN set -eux; \
    base_prefix_found=0; \
    for base_python in /opt/conda/bin/python /opt/miniconda/bin/python /opt/conda/envs/ptca/bin/python; do \
        if [ -x "$base_python" ]; then \
            base_prefix_found=1; \
            "$base_python" -m pip install --no-cache-dir --progress-bar off --upgrade \
                'pip>=26.1.2' \
                'PyJWT>=2.13.0' \
                'idna>=3.15' \
                'py-rattler>=0.24.0' \
                'cryptography>=48.0.1'; \
        fi; \
    done; \
    test "$base_prefix_found" = "1"; \
    conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --progress-bar off --upgrade 'pip>=26.1.2'

# Install packages via pip (avoids conda solver OOM)
RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --progress-bar off \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'setuptools-git' \
    'setuptools==81.0.0' \
    'psutil>5.0.0,<6.0.0' \
    'torch==2.10.0'

# Install pip dependencies
RUN pip install --progress-bar off \
                azureml-core==1.61.0.post4 \
                azureml-mlflow==1.62.0.post5 \
                azureml-pipeline-core==1.62.0 \
                azureml-telemetry==1.62.0 \
                azureml-interpret==1.62.0 \
                azureml-responsibleai==1.62.0 \
                azureml-automl-core==1.62.0.post3 \
                azureml-automl-runtime==1.62.0.post1 \
                azureml-train-automl-client==1.62.0 \
                azureml-train-automl-runtime==1.62.0 \
                azureml-train-automl==1.62.0 \
                azureml-dataset-runtime==1.62.0 \
                azureml-defaults==1.62.0 \
                'mlflow-skinny==2.16.0' \
                'xgboost==1.5.2' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'pillow==12.2.0'

# Install pip ad-hoc dependencies for security updates
# distributed>=2026.1.0 (CVE-2026-23528): XSS leading to RCE via Dask dashboard
#   Chain: azureml-train-automl-runtime -> dask[complete] -> distributed
#   Parent (azureml-train-automl-runtime==1.62.0, latest) still pins dask[complete]<=2023.2.0;
#   parent fix not available, override required.
#
# bokeh>=3.8.2 (GHSA-793v-589g-574v): conda env installs 2.4.3, pip can't auto-upgrade
#   Chain: azureml-train-automl-runtime -> bokeh
#   Parent (azureml-train-automl-runtime==1.62.0, latest) still pins bokeh<3.0.0;
#   parent fix not available, override required.
#
# python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94): set_key()/unset_key() follow symlinks
#   on cross-device .env writes, allowing arbitrary file overwrite.
#   Pulled in transitively (no direct top-level package declares it; latest
#   mlflow-skinny declares python-dotenv<2,>=0.19.0 but the pinned mlflow-skinny==2.16.0
#   does not; it is brought in by another transitive path). No parent constraint
#   blocks 1.2.2, so override directly.
#
# pyarrow>=23.0.1 (GHSA-rgxp-2hwp-jwgg / CVE-2026-25087): use-after-free
#   in Apache Arrow C++. Parent packages still resolve pyarrow 17.0.0 in this
#   image through AzureML AutoML runtime dependencies, so override directly until
#   upstream pins move to a patched version.
# cryptography>=48.0.1 (GHSA-537c-gmf6-5ccf): pinned transitive dep of
#   azureml-core/azureml-automl packages; fixes vulnerable cryptography 46.0.7
#   resolved in the AzureML AutoML environment.
# onnx: azureml-automl-runtime pins onnx<=1.17.0; override needed for
#   GHSA-3r9x-f23j-gc73, GHSA-p433-9wv8-28xj, GHSA-q56x-g2fj-4rj6,
#   GHSA-538c-55jv-c5g9, GHSA-cmw6-hcpp-c6jp, GHSA-hqmj-h5c6-369m
# pillow: upgraded from 12.1.1 for GHSA-whj4-6x5x-4v2j
RUN pip install --no-cache-dir --progress-bar off --upgrade \
    'distributed>=2026.1.0' \
    'bokeh>=3.8.2' \
    'cryptography>=48.0.1' \
    'onnx>=1.21.0' \
    'pillow>=12.2.0' \
    'python-dotenv>=1.2.2' \
    'pyarrow>=23.0.1'  # pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg / CVE-2026-25087
RUN pip install --no-deps --force-reinstall 'skl2onnx==1.19.1'
RUN pip install --no-deps --force-reinstall 'onnxconverter-common==1.16.0'
RUN pip install --no-deps --force-reinstall 'onnxmltools==1.14.0'
RUN pip install --no-deps --force-reinstall 'pandas==2.0.0'
COPY patch_automl_pandas2_sparse.py /tmp/patch_automl_pandas2_sparse.py
RUN python /tmp/patch_automl_pandas2_sparse.py && \
    rm /tmp/patch_automl_pandas2_sparse.py
