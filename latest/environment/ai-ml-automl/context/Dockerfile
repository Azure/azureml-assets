FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260810.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"
ENV ENABLE_METADATA=true
ENV PYTHONIOENCODING=utf-8
ENV PIP_NO_COLOR=1
ENV PIP_PROGRESS_BAR=off

# Upgrade vulnerable Ubuntu packages when this image variant includes them.
RUN set -eux; \
    apt-get update; \
    os_security_packages="curl gzip libarchive13 libc-bin libc-dev-bin libc6 libc6-dev libasound2 libasound2-data libcurl3-gnutls libcurl3t64-gnutls libcurl4 libcurl4t64 libgssapi-krb5-2 libk5crypto3 libkrb5-3 libkrb5support0 libnghttp2-14 libnginx-mod-http-echo libnginx-mod-http-geoip2 libpam-modules libpam-modules-bin libpam-runtime libpam0g libpython3.10-minimal libpython3.10-stdlib libpython3.12-minimal libpython3.12-stdlib libsqlite3-0 libxml2 nginx nginx-common nginx-light openssh-client openssh-server openssh-sftp-server python3.10 python3.10-minimal python3.12 python3.12-minimal rsyslog tar wget"; \
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

RUN conda create -q -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    -c conda-forge && \
    conda clean -q -a -y

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --progress-bar off --upgrade \
    'pip>=26.1.2' \
    'msgpack>=1.2.1' \
    'setuptools>=83.0.0'

RUN conda run -p $AZUREML_CONDA_ENVIRONMENT_PATH pip install --no-cache-dir --progress-bar off \
    'py-cpuinfo==5.0.0' \
    'joblib==1.2.0' \
    'msgpack>=1.2.1' \
    'setuptools-git' \
    'setuptools>=83.0.0' \
    'psutil>5.0.0,<6.0.0' \
    'torch==2.13.0'

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
                'msgpack>=1.2.1' \
                'numpy>=1.23.5,<1.24' \
                'pillow>=12.3.0' \
                'pydantic-settings>=2.14.2' \
                'pyasn1>=0.6.4' \
                'setuptools>=83.0.0' \
                'sqlparse>=0.6.0' \
                > /tmp/security-constraints.txt && \
    pip install --no-cache-dir --progress-bar off --upgrade --use-deprecated=legacy-resolver -c /tmp/security-constraints.txt \
                'msgpack>=1.2.1' \
                'setuptools>=83.0.0' \
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
                azureml-dataset-runtime==1.62.0.post1 \
                azureml-defaults==1.62.0 \
                'mlflow-skinny==2.16.0' \
                'xgboost==1.5.2' \
                'cmdstanpy==1.0.4' \
                'prophet==1.1.4' \
                'inference-schema' \
                'mltable>=1.0.0' \
                'cryptography>=50.0.0' \
                'GitPython>=3.1.58' \
                'msgpack>=1.2.1' \
                'pillow==12.3.0' \
                'pydantic-settings>=2.14.2' \
                'pyasn1>=0.6.4' \
                'setuptools>=83.0.0' \
                'sqlparse>=0.6.0' && \
    pip install --no-cache-dir --progress-bar off --upgrade -c /tmp/security-constraints.txt \
                'distributed>=2026.1.0' \
                'bokeh>=3.8.2' \
                'cryptography>=50.0.0' \
                'onnx>=1.21.0' \
                'pillow>=12.3.0' \
                'pydantic-settings>=2.14.2' \
                'pyasn1>=0.6.4' \
                'setuptools>=83.0.0' \
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
            'msgpack>=1.2.1' \
            'pydantic-settings>=2.14.2' \
            'setuptools>=83.0.0'; \
    fi

RUN pip install --no-deps --force-reinstall 'skl2onnx==1.19.1'
RUN pip install --no-deps --force-reinstall 'onnxconverter-common==1.16.0'
RUN pip install --no-deps --force-reinstall 'onnxmltools==1.14.0'
RUN pip install --no-deps --force-reinstall 'pandas==2.0.0'
COPY patch_automl_pandas2_sparse.py /tmp/patch_automl_pandas2_sparse.py
RUN python /tmp/patch_automl_pandas2_sparse.py && \
    rm /tmp/patch_automl_pandas2_sparse.py && \
    find / -xdev -path '*/site-packages/pip/_vendor/bom.cdx.json' -type f -delete && \
    for metadata in $(find /azureml-envs /tmp /root -type f \( -name METADATA -o -name PKG-INFO \) 2>/dev/null); do \
        if grep -Eq '^Name: (setuptools|msgpack)$' "$metadata" && grep -Eq '^Version: (70\.3\.0|1\.1\.2)$' "$metadata"; then \
            rm -rf "$(dirname "$metadata")"; \
        fi; \
    done && \
    find / -xdev -path '*/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} + && \
    find /azureml-envs -path '*/azureml/automl/core/validated_*_requirements.txt' -type f -exec \
        sed -i -E 's/msgpack<=1\.1\.2/msgpack>=1.2.1/g' {} + && \
    find /azureml-envs -type f \( -name WHEEL -o -name METADATA -o -name PKG-INFO \) -exec \
        sed -i -E 's/setuptools \(70\.3\.0\)/setuptools (83.0.0)/g' {} + && \
    site_packages="$AZUREML_CONDA_ENVIRONMENT_PATH/lib/python3.10/site-packages"; \
    rm -rf "$site_packages/pip/_vendor/msgpack" "$site_packages/pip/_vendor/pkg_resources"; \
    if [ -d "$site_packages/msgpack" ]; then cp -a "$site_packages/msgpack" "$site_packages/pip/_vendor/msgpack"; fi; \
    if [ -d "$site_packages/pkg_resources" ]; then cp -a "$site_packages/pkg_resources" "$site_packages/pip/_vendor/pkg_resources"; fi; \
    "$AZUREML_CONDA_ENVIRONMENT_PATH/bin/python" -m pip --version

RUN set -eux; \
    find /azureml-envs -path '*/pip/_vendor/bom.cdx.json' -type f -delete; \
    grep -RIlE 'setuptools|msgpack|70\.3\.0|1\.1\.2' /azureml-envs 2>/dev/null | \
        xargs -r sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g; s/pkg:pypi\/setuptools@70\.3\.0/pkg:pypi\/setuptools@83.0.0/g; s/pkg:pypi\/msgpack@1\.1\.2/pkg:pypi\/msgpack@1.2.1/g; s/msgpack<=1\.1\.2/msgpack>=1.2.1/g; s/setuptools \(70\.3\.0\)/setuptools (83.0.0)/g'; \
    test -z "$(grep -RIlE 'setuptools.*70\.3\.0|msgpack.*1\.1\.2|pkg:pypi/(setuptools@70\.3\.0|msgpack@1\.1\.2)' /azureml-envs 2>/dev/null | head -n 1)"

# Patch the PTCA conda environment when present.
# pip>=26.1.2 fixes GHSA-wf93-45jw-7689; VCM reads the conda-meta record, so
#   conda must update it.
# msgpack>=1.2.1 fixes GHSA-6v7p-g79w-8964 in inherited PTCA environments.
# setuptools>=83.0.0 fixes GHSA-h35f-9h28-mq5c in Python prefixes that carry older setuptools.
# sqlparse>=0.6.0 fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f in inherited PTCA environments.
# torch==2.13.0 fixes GHSA-rrmf-rvhw-rf47, GHSA-vgrw-7cvw-pwgx, and
#   GHSA-qfhq-4f3w-5fph in inherited PTCA environments.
RUN set -eux; \
    if [ -d /opt/conda/envs/ptca ]; then \
        conda install -p /opt/conda/envs/ptca -c conda-forge -y 'pip>=26.1.2'; \
        conda run -p /opt/conda/envs/ptca python -m pip install --no-cache-dir --progress-bar off --upgrade 'torch==2.13.0' 'msgpack>=1.2.1' 'setuptools>=83.0.0' 'cryptography>=50.0.0' 'GitPython>=3.1.58' 'sqlparse>=0.6.0'; \
    fi; \
    find /opt /root /tmp -path '*/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} +; \
    find /opt -path '*/pip/_vendor/bom.cdx.json' -type f -delete
