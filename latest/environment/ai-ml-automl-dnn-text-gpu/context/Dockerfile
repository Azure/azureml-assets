FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-text-gpu
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default && \
    apt-get autoremove -y

# Security: upgrade Ubuntu packages to patched versions.
# USN-8298-1: dotnet-{hostfxr,host,runtime}-8.0 8.0.26-0ubuntu1~22.04.1 -> 8.0.27-0ubuntu1~22.04.1
# USN-8319-1: libgcrypt20 1.9.4-3ubuntu3 -> 1.9.4-3ubuntu3.2
# USN-8354-1: nginx-common/nginx-light/libnginx-mod-* 1.18.0-6ubuntu14.11 -> 1.18.0-6ubuntu14.12
# USN-8362-1: liblzma5/xz-utils 5.2.5-2ubuntu1 -> 5.2.5-2ubuntu1.1
# `--only-upgrade` is a no-op for packages not present, so this is safe even if
# dotnet is absent from the base image.
RUN apt-get update && \
    apt-get install -y --only-upgrade \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        libgcrypt20 \
        nginx-common \
        libnginx-mod-http-geoip2 \
        nginx-light \
        libnginx-mod-http-echo \
        liblzma5 \
        xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# Security: upgrade pip in /opt/conda (base, py3.13) and /opt/conda/envs/ptca (py3.10)
# to fix CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9 (vulnerable self-update behaviour in
# pip < 26.1). Current base tag biweekly.202605.2 ships pip==26.0.1 in both
# envs. pip is its own parent, so no upstream package can ship a fixed pip.
# Keep this as its own RUN to avoid the documented pip self-upgrade race.
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade 'pip>=26.1' && \
    /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'pip>=26.1'

# begin conda create
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    # begin conda dependencies
    pip \
    numpy~=1.23.5\
    scikit-learn=1.5.1 \
    pandas~=1.5.3 \
    setuptools=82.0.1 \
    wheel=0.46.3 \
    scipy=1.10.1 \
    pybind11=2.10.1 \
    # end conda dependencies
    -c conda-forge -c anaconda

# Ensure additional conda and pip install commands apply to our conda env of interest.
SHELL ["conda", "run", "-p", "$AZUREML_CONDA_ENVIRONMENT_PATH", "/bin/bash", "-c"]

# begin pip install
# Install pip dependencies
# Here, we pin sentencepiece since 0.1.98 breaks training. Earlier versions of horovod contain a sev2 vulnerability,
# and earlier versions of tokenizers cause log spam with transformers==4.16.0.
RUN pip install --no-cache-dir \
                # begin pypi dependencies
                azureml-core==1.61.0.post4 \
                azureml-mlflow==1.62.0.post5 \
                azureml-automl-core==1.62.0.post3 \
                azureml-automl-runtime==1.62.0.post1 \
                azureml-defaults==1.62.0 \
                'azure-identity>=1.25.1' \
                'sentencepiece==0.2.1' \
                'filelock>=3.20.3'
                # end pypi dependencies

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client==1.62.0 \
                azureml-train-automl-runtime==1.62.0 \
                azureml-automl-dnn-nlp==1.62.0

# Separate updates for fixing vulnerabilities.
# Doing it  separately from pip install above to avoid conflict with other packages
# We should aim for this list to be empty with new and patched releases
# by fixing dependencies in the base packages
RUN pip install pyarrow==14.0.2 \
                accelerate==1.12.0 

# Override transformers to fix GHSA-69w3-r845-3855
# Root cause: azureml-automl-dnn-nlp==1.62.0 (latest as of 2026-05-26)
# pins transformers==4.53.0; cannot upgrade parent.
RUN pip install --no-cache-dir --no-deps 'transformers[sentencepiece,torch]==5.5.4'

# Security: upgrade pip to fix CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9 (vulnerable
# self-update behaviour in pip < 26.1) in the AZUREML conda env. The `conda create`
# above seeds the env with pip==26.0.1 from conda-forge. pip is its own parent,
# so no upstream package can bring in a patched pip; an explicit upgrade is the
# only remediation. Kept as its own RUN to avoid the pip self-upgrade race.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1'

# Upgrade bokeh, cryptography, and onnx in the AZUREML conda env (py3.10).
# NOTE: azureml-mlflow==1.62.0.post2 (latest as of 2026-05-26) caps
# cryptography<47.0.0 but does not force the fixed floor; use >=46.0.5.
# Override onnx to fix GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6, GHSA-p433-9wv8-28xj, GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m
# Root cause: azureml-automl-runtime==1.62.0 (latest) pins onnx<=1.17.0; cannot upgrade parent
RUN pip install --upgrade 'distributed>=2026.1.0' 'cryptography>=46.0.5' 'bokeh>=3.8.2' 'onnx>=1.21.0'

# Fix vulnerable vendored dependencies in base and ptca setuptools.
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced.
# Current base tag biweekly.202605.2 ships: base=82.0.0, ptca=81.0.0.
# Both are below 82.0.1, so direct overrides are still required.
RUN /opt/conda/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1' && \
    /opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

# Security: upgrade urllib3 in the base env (py3.13) to fix
# CVE-2026-44431 / GHSA-qccp-gfcp-xxvc and CVE-2026-44432 / GHSA-mf9v-mfxr-j63j
# (urllib3 < 2.7.0). Current base tag biweekly.202605.2 ships urllib3 2.6.3
# in base; ptca already ships urllib3 2.7.0, so no ptca override is needed.
# Latest parents (`requests==2.34.2`, `distributed==2026.3.0`) still use loose
# urllib3 ranges, so parent upgrades cannot force urllib3>=2.7.0.
#
# Security: upgrade idna in the base env (py3.13) to fix CVE-2026-45409 /
# GHSA-65pc-fj4g-8rjx (idna < 3.15). Current base tag biweekly.202605.2 ships
# idna 3.11 in base; ptca already ships idna 3.15. Latest parents
# (`requests==2.34.2`, `anyio==4.13.0`, `httpx==0.28.1`, `yarl==1.24.2`)
# still use loose idna ranges, so parent upgrades cannot force idna>=3.15.
#
# Security: upgrade click in base env (py3.13) to fix CVE-2026-7246 /
# GHSA-47fr-3ffg-hgmw (click.edit() command injection in click < 8.3.3).
# Current base tag biweekly.202605.2 ships click 8.2.1. Latest parents
# (`typer==0.25.1`, `anaconda-cli-base==0.8.2`, and httpx's CLI extra)
# still use loose click ranges, so parent upgrades cannot force click>=8.3.3.
#
# PyJWT>=2.13.0: pinned transitive dep of ACPT base image; fixes
# GHSA-993g-76c3-p5m4 and GHSA-jq35-7prp-9v3f.
#
# py-rattler>=0.24.0: pinned transitive dep of ACPT base image; fixes
# GHSA-q53q-5r4j-5729.
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade \
    'urllib3>=2.7.0' \
    'idna>=3.15' \
    'click>=8.3.3' \
    'PyJWT>=2.13.0' \
    'py-rattler>=0.24.0'

# Security: python-dotenv 1.2.1 -> >=1.2.2 fixes GHSA-mf9w-mj56-hr94 (set_key()/
# unset_key() follow symlinks on cross-device .env writes -> arbitrary file
# overwrite). Lives in /opt/conda/lib/python3.13/site-packages of the base conda
# env (shipped by the current ACPT base image biweekly.202605.2). It is pulled in
# transitively by anaconda-auth (Requires-Dist: python-dotenv with no version pin)
# and pydantic-settings (python-dotenv>=0.21.0, via anaconda-cli-base ->
# anaconda-auth). Latest releases on PyPI as of 2026-05-26 (anaconda-auth==0.15.0,
# pydantic-settings==2.14.1) still use the same loose floors, so a parent upgrade
# cannot force >=1.2.2; direct override is the only remediation.
RUN conda run -n base pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'

# Security: upgrade aiohttp to fix GHSA-hg6j-4rv6-33pg and GHSA-jg22-mg44-37j8
# (aiohttp < 3.14.0). Installed in three Python environments:
#   1. azureml-automl-dnn-text-gpu conda env (py3.10) — via SHELL (conda run)
#   2. /opt/conda base env (py3.13)
#   3. /opt/conda/envs/ptca env (py3.10)
# Latest parents (aiohttp is not directly pinned by any upper constraint) allow
# the upgrade; no --no-deps needed.
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0'
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' && \
    /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0'

# Apply the same setuptools vendored-dependency fix in the AzureML conda env.
RUN pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

RUN /bin/bash -c "source activate $AZUREML_CONDA_ENVIRONMENT_PATH && \
 export CUDACXX=/usr/local/cuda/bin/nvcc && \
 export HOROVOD_BUILD_CUDA_CC_LIST='60,61,70,75,80,86,89,90' && \
 HOROVOD_WITH_PYTORCH=1 \
 HOROVOD_CUDA_HOME=/usr/local/cuda \
 CMAKE_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
 pip install --no-cache-dir --no-build-isolation \
 git+https://github.com/horovod/horovod@3a31d933a13c7c885b8a673f4172b17914ad334d"

RUN rm -rf /opt/conda/pkgs/
# end pip install
