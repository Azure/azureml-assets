FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}

# IMPORTANT (2026-10-26): Python was upgraded from 3.10 to 3.11 for Pillow 13 security fixes.
# Downstream users must update interpreter-specific paths, wheels, and compiled dependencies for Python 3.11.

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
ENV ENABLE_METADATA=true
ENV PYTHONIOENCODING=utf-8
ENV PIP_NO_COLOR=1
ENV PIP_PROGRESS_BAR=off

RUN conda create -q -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.11 \
    -c conda-forge && \
    conda clean -q -a -y

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --progress-bar off --upgrade \
    'pip>=26.2.0' \
    'msgpack>=1.2.3' \
    'setuptools>=84.1.0'

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --progress-bar off \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'msgpack>=1.2.3' \
    'setuptools-git' \
    'setuptools>=84.1.0' \
    'psutil>5.0.0,<6.0.0' \
    'torch==2.13.0'

# azureml-automl-runtime pins pmdarima 1.8.5, whose legacy build metadata omits
# NumPy and no longer builds in an isolated environment with current setuptools.
# Build it first with explicit build dependencies, then restore patched setuptools.
RUN pip install --no-cache-dir --progress-bar off --upgrade \
        'setuptools==59.8.0' 'Cython<3' 'numpy>=1.23.5,<1.24' wheel && \
    pip install --no-cache-dir --progress-bar off --no-deps --no-build-isolation \
        'pmdarima==1.8.5' && \
    pip install --no-cache-dir --progress-bar off --upgrade 'setuptools>=84.1.0'

# pmdarima 1.8.5 imports pkg_resources, which patched setuptools no longer ships.
RUN find $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.11/site-packages/pmdarima -type f -name '*.py' \
        -exec sed -i 's/from pkg_resources import parse_version/from packaging.version import parse as parse_version/' {} + && \
    ! grep -R 'from pkg_resources import parse_version' \
        $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.11/site-packages/pmdarima

# Install pip dependencies, then apply security overrides for vulnerable transitive packages.
# distributed and bokeh are pinned transitive deps of azureml-train-automl-runtime; fixes CVE-2026-23528 and GHSA-793v-589g-574v.
# cryptography is a pinned transitive dep of AzureML/MLflow and the base image; fixes GHSA-g6cj-pr64-35w5.
# GitPython is a pinned transitive dep of MLflow; fixes GHSA-3f7w-8rr8-f37f, GHSA-539m-9xh6-q6rr, GHSA-p538-c434-8v24, GHSA-wvpp-8hx9-p66j, GHSA-9rj7-rf2p-w77r, GHSA-4gmw-gg2m-w46p, GHSA-hh9p-6wh2-4mfc, GHSA-jm78-9fvv-mhgr, and GHSA-hmq2-w58f-27jc.
# msgpack is a pinned transitive dep of the AzureML AutoML Dask stack; fixes GHSA-6v7p-g79w-8964.
# onnx is pinned because azureml-automl-runtime constrains onnx<=1.17.0; fixes current ONNX GHSA findings.
# pyarrow is a pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg / CVE-2026-25087.
# setuptools is kept above vulnerable versions in Python prefixes; fixes GHSA-h35f-9h28-mq5c.
# sqlparse is a pinned transitive dep of mlflow-skinny; fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f.
# numpy>=1.23.5,<1.24 is NOT a security floor. onnx pulls ml_dtypes, which as of 0.6.0
# requires numpy 2.x. Without the cap in the same resolve as the security floors, pip
# silently upgrades numpy to 2.2.6, which contradicts azureml-automl-runtime and leaves the
# pandas 2.0.0 wheel force-reinstalled below binary-incompatible, so the environment fails
# on `import pandas` at runtime ("ValueError: numpy.dtype size changed"). Matches the cap
# already applied in ai-ml-automl-dnn.
RUN printf '%s\n' \
                'cryptography>=50.0.0' \
                'GitPython>=3.1.58' \
                'msgpack>=1.2.3' \
                'numpy>=1.23.5,<1.24' \
                'pillow>=13.0.0' \
                'pydantic-settings>=2.14.2' \
                'pyasn1>=0.6.4' \
                'setuptools>=84.1.0' \
                'sqlparse>=0.6.0' \
                > /tmp/security-constraints.txt && \
    pip install --no-cache-dir --progress-bar off --upgrade --use-deprecated=legacy-resolver -c /tmp/security-constraints.txt \
                'msgpack>=1.2.3' \
                'setuptools>=84.1.0' \
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
                azureml-train-automl=={{latest-pypi-version}} \
                azureml-dataset-runtime=={{latest-pypi-version}} \
                azureml-defaults=={{latest-pypi-version}} \
                'mlflow-skinny==2.16.0' \
                'xgboost==1.5.2' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'cryptography>=50.0.0' \
                'GitPython>=3.1.58' \
                'msgpack>=1.2.3' \
                'pillow==13.0.0' \
                'pydantic-settings>=2.14.2' \
                'pyasn1>=0.6.4' \
                'setuptools>=84.1.0' \
                'sqlparse>=0.6.0' && \
    pip install --no-cache-dir --progress-bar off --upgrade -c /tmp/security-constraints.txt \
                'distributed>=2026.1.0' \
                'bokeh>=3.8.2' \
                'cryptography>=50.0.0' \
                'onnx>=1.21.0' \
                'pillow>=13.0.0' \
                'pydantic-settings>=2.14.2' \
                'pyasn1>=0.6.4' \
                'setuptools>=84.1.0' \
                'python-dotenv>=1.2.2' \
                'pyarrow>=23.0.1' \
                'sqlparse>=0.6.0' && \
    conda clean -q -a -y && \
    rm -rf /root/.cache/pip /tmp/pip-* /opt/conda/pkgs /opt/miniconda/pkgs /tmp/security-constraints.txt

# Patch vulnerable Python packages inherited in the base Miniconda prefix.
# These are transitive base-image packages, not direct dependencies of this image:
# cryptography fixes GHSA-g6cj-pr64-35w5; h2 fixes GHSA-6hr6-w5qg-qmwg;
# msgpack fixes GHSA-6v7p-g79w-8964; pydantic-settings fixes GHSA-4xgf-cpjx-pc3j;
# setuptools fixes GHSA-5rjg-fvgr-3xxf and GHSA-h35f-9h28-mq5c.
RUN if [ -x /opt/miniconda/bin/python ]; then \
        /opt/miniconda/bin/python -m pip install --no-cache-dir --progress-bar off --upgrade \
            'cryptography>=50.0.0' \
            'h2>=4.4.1' \
            'msgpack>=1.2.3' \
            'pydantic-settings>=2.14.2' \
            'setuptools>=84.1.0'; \
    fi

RUN pip install --no-deps --force-reinstall 'skl2onnx==1.19.1'
RUN pip install --no-deps --force-reinstall 'onnxconverter-common==1.16.0'
RUN pip install --no-deps --force-reinstall 'onnxmltools==1.14.0'
RUN pip install --no-deps --force-reinstall 'pandas==2.0.0'
COPY patch_automl_pandas2_sparse.py /tmp/patch_automl_pandas2_sparse.py
RUN python /tmp/patch_automl_pandas2_sparse.py && \
    rm /tmp/patch_automl_pandas2_sparse.py

# Patch the PTCA conda environment when present.
# pip>=26.2.0 fixes GHSA-qwm4-qh6w-59xr; VCM reads the conda-meta record, so
#   conda must update it.
# msgpack>=1.2.3 fixes GHSA-6v7p-g79w-8964 in inherited PTCA environments.
# setuptools>=84.1.0 fixes GHSA-h35f-9h28-mq5c in Python prefixes that carry older setuptools.
# sqlparse>=0.6.0 fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f in inherited PTCA environments.
# torch==2.13.0 fixes GHSA-rrmf-rvhw-rf47, GHSA-vgrw-7cvw-pwgx, and
#   GHSA-qfhq-4f3w-5fph in inherited PTCA environments.
RUN set -eux; \
    if [ -d /opt/conda/envs/ptca ]; then \
        conda install -p /opt/conda/envs/ptca -c conda-forge -y 'pip>=26.2.0'; \
        conda run -p /opt/conda/envs/ptca python -m pip install --no-cache-dir --progress-bar off --upgrade 'torch==2.13.0' 'msgpack>=1.2.3' 'setuptools>=84.1.0' 'cryptography>=50.0.0' 'GitPython>=3.1.58' 'sqlparse>=0.6.0'; \
    fi

# Normalize pip's vendored inventory after the corresponding packages are upgraded.
RUN find /opt /azureml-envs /root -path '*/pip/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==84.1.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} + && \
    find /opt /azureml-envs /root -path '*/pip/_vendor/bom.cdx.json' -type f -delete

# XGBoost 1.5.2 uses pkg_resources only to detect optional Dask support.
RUN sed -i \
        -e 's/import pkg_resources/from importlib import metadata as importlib_metadata/' \
        -e 's/pkg_resources\.get_distribution/importlib_metadata.distribution/g' \
        -e 's/pkg_resources\.DistributionNotFound/importlib_metadata.PackageNotFoundError/g' \
        $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.11/site-packages/xgboost/compat.py && \
    ! grep -q 'pkg_resources' \
        $AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.11/site-packages/xgboost/compat.py

# Reassert newly patched transitive dependencies after all package installs.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.15.0'

# Reassert November 2026 vulnerability-report floors in the scanned Python prefixes.
RUN /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'certifi>=2026.10.19'

# Reassert November 2026 OS-package floors for installed packages only.
RUN set -eux; \
    apt-get update; \
    security_packages="nginx libbrotli1 libpcre2-8-0"; \
    installed_packages=""; \
    for package in $security_packages; do \
        if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then \
            installed_packages="$installed_packages $package"; \
        fi; \
    done; \
    if [ -n "$installed_packages" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --only-upgrade $installed_packages; \
    fi; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*
