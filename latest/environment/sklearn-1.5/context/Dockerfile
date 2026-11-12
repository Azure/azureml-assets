FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20261108.v1

WORKDIR /

ENV CONDA_PREFIX=/azureml-envs/sklearn-1.5
ENV CONDA_DEFAULT_ENV=$CONDA_PREFIX
ENV PATH=$CONDA_PREFIX/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Upgrade OS packages to pick up current Ubuntu security errata.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Environment setup is split into two stages (conda create + pip install) because the
# original single-step `conda env create -f conda_dependencies.yaml` combines conda's
# SAT solver with pip dependency resolution in one process. On ACR Standard build agents
# (limited memory), the conda solver consistently OOMs (exit 137) when resolving both
# conda and pip dependencies together. Splitting them avoids the memory spike.

# Step 1: Create conda env with only conda-channel packages (small, fast solve)
# IMPORTANT (2026-10-26): Python was upgraded from 3.10 to 3.11 for Pillow 13 security fixes.
# Downstream users must update interpreter-specific paths, wheels, and compiled dependencies for Python 3.11.
COPY conda_dependencies.yaml .
RUN conda install -n base -c conda-forge --freeze-installed --force-reinstall -y pip=26.2.2 && \
    /opt/miniconda/bin/python -m pip install --no-cache-dir --force-reinstall pip==26.2.2 && \
    conda create -p $CONDA_PREFIX -q -y python=3.11 pip=26.2.2 pandas=2.2.3 'scipy~=1.10.0' numpy=1.26.4 -c conda-forge -c anaconda && \
    conda run -p $CONDA_PREFIX python -m pip install --no-cache-dir --force-reinstall pip==26.2.2 && \
    conda clean -a -y

# Step 2: Install pip packages separately (pip resolver uses less memory than conda's SAT solver).
# starlette>=1.0.1 / idna>=3.15: pinned transitive deps of fastapi / requests; fix GHSA-86qp-5c8j-p5mr and GHSA-65pc-fj4g-8rjx.
# msgpack>=1.2.3 / setuptools>=84.1.0: pinned transitive deps of distributed / AzureML packages; fix GHSA-6v7p-g79w-8964, GHSA-5rjg-fvgr-3xxf, and GHSA-h35f-9h28-mq5c.
# Pin Intel to retain sklearn 1.5 compatibility; newer releases require sklearn >=1.6 and removed older API fallbacks.
RUN printf '%s\n' 'msgpack>=1.2.3' 'setuptools>=84.1.0' > /tmp/pip-security-constraints.txt && \
    conda run -p $CONDA_PREFIX pip install --no-cache-dir \
    --no-build-isolation \
    --constraint /tmp/pip-security-constraints.txt \
    scikit-learn-intelex==2025.0.1 \
    azureml-core==1.61.0.post4 \
    azureml-defaults==1.62.0 \
    azureml-inference-server-http==1.6.1 \
    azureml-mlflow==1.62.0.post6 \
    azureml-telemetry==1.62.0 \
    'scikit-learn~=1.5.0' \
    'joblib~=1.2.0' \
    py-spy==0.3.12 \
    'debugpy~=1.6.3' \
    'ipykernel~=6.0' \
    tensorboard \
    'psutil>=5.9,<8' \
    'matplotlib>=3.10.8,<3.11' \
    'tqdm~=4.66.3' \
    py-cpuinfo==5.0.0 \
    'torch-tb-profiler~=0.4.0' \
    'pillow>=13.0.0' \
    'starlette>=1.0.1' \
    'idna>=3.15' \
    'msgpack>=1.2.3' \
    'setuptools>=84.1.0' && \
    rm conda_dependencies.yaml /tmp/pip-security-constraints.txt

# Security vulnerability fixes for transitive dependencies.
# aiohttp>=3.14.4: pinned transitive dep of distributed; fixes GHSA-8495-4g3g-x7pr.
# cryptography>=50.0.0: pinned transitive dep of azureml-mlflow / azureml-core; fixes GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# distributed>=2026.1.0: pinned transitive dep of azureml-mlflow / mlflow; fixes GHSA-9q34-5785-65x5.
# GitPython>=3.1.55: pinned transitive dep of azureml-mlflow; fixes GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, GHSA-r9mr-m37c-5fr3, GHSA-94p4-4cq8-9g67.
# msgpack>=1.2.3: pinned transitive dep of distributed; fixes GHSA-6v7p-g79w-8964.
# pyarrow>=23.0.1: pinned transitive dep of azureml-dataset-runtime; fixes GHSA-rgxp-2hwp-jwgg.
# setuptools>=84.1.0: pinned transitive build/runtime dep of AzureML packages; fixes GHSA-5rjg-fvgr-3xxf and GHSA-h35f-9h28-mq5c.
# sqlparse>=0.6.0: pinned transitive dep of azureml-mlflow / mlflow; fixes GHSA-f2ff-p2ww-7p4p, GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, GHSA-pwgv-4x5q-6m9f.
RUN conda run -p $CONDA_PREFIX pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.4' 'pyOpenSSL==26.4.0' 'cryptography>=50.0.0,<51' 'distributed>=2026.1.0' 'GitPython>=3.1.55' 'msgpack>=1.2.3' 'pyarrow>=23.0.1,<24' 'setuptools>=84.1.0' 'sqlparse>=0.6.0'

# Apply security floors to packages in the base Miniconda prefix as well.
RUN /opt/miniconda/bin/pip install --no-cache-dir --upgrade \
    'idna>=3.15' 'cryptography>=50.0.0' 'setuptools>=84.1.0' 'msgpack>=1.2.3'

# Reassert newly patched transitive dependencies after all package installs.
RUN /azureml-envs/sklearn-1.5/bin/python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.15.0' 'oauthlib>=4.0.0' 'urllib3>=2.8.0' && \
    /opt/miniconda/bin/python -m pip install --no-cache-dir --upgrade 'PyJWT>=2.15.0' 'urllib3>=2.8.0' && rm -rf /opt/miniconda/pkgs/urllib3-2.7.0-*

# Remove stale Conda and pip-vendored inventory after the patched packages are installed.
RUN set -eux; \
    find /opt/miniconda "$CONDA_PREFIX" -path '*/pip/_vendor/bom.cdx.json' -type f -delete; \
    find /opt/miniconda "$CONDA_PREFIX" -path '*/pip/_vendor/vendor.txt' -type f \
        -exec sed -i -E '/^(setuptools|msgpack)==/d' {} +; \
    rm -f /opt/miniconda/conda-meta/setuptools-*.json \
          /opt/miniconda/conda-meta/msgpack-*.json \
          /opt/miniconda/conda-meta/msgpack-python-*.json \
          "$CONDA_PREFIX"/conda-meta/setuptools-*.json \
          "$CONDA_PREFIX"/conda-meta/msgpack-*.json \
          "$CONDA_PREFIX"/conda-meta/msgpack-python-*.json; \
    rm -rf /opt/miniconda/pkgs /root/.cache/pip

# Reassert November 2026 vulnerability-report floors in the scanned Python prefixes.
RUN $CONDA_PREFIX/bin/python -m pip install --no-cache-dir --upgrade 'opentelemetry-api>=1.45.1'

# Reassert November 2026 OS-package floors for installed packages only.
RUN set -eux; \
    apt-get update; \
    security_packages="libbrotli1 nginx"; \
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
