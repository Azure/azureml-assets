FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/azureml-automl-dnn-vision-gpu
# Prepend path to AzureML conda environment
ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

COPY --from=mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20250506.v1 /var/mlflow_resources/ /var/mlflow_resources/

ENV MLFLOW_MODEL_FOLDER="mlflow-model"

# Inference requirements
# Security: reinstall packages to pick up patched versions even when an older base layer
# is cached (apt-get upgrade alone has been observed to leave held packages in place).
# USN-8298-1: dotnet-host/hostfxr/runtime-8.0 8.0.26 -> 8.0.27
# USN-8319-1: libgcrypt20 1.9.4-3ubuntu3 -> 1.9.4-3ubuntu3.2
# USN-8354-1: nginx-common/light/libnginx-mod-* 1.18.0-6ubuntu14.11 -> 1.18.0-6ubuntu14.12
# USN-8362-1: liblzma5/xz-utils 5.2.5-2ubuntu1 -> 5.2.5-2ubuntu1.1
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20250310.v1 /artifacts /var/
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        liblttng-ust1 \
        libunwind8 \
        libxml++2.6-2v5 \
        nginx-light \
        psmisc \
        rsyslog \
        runit \
        unzip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --reinstall -y \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        libgcrypt20 \
        liblzma5 \
        xz-utils \
        nginx-common \
        nginx-light \
        libnginx-mod-http-geoip2 \
        libnginx-mod-http-echo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default && \
    apt-get autoremove -y

ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

ENV ENABLE_METADATA=true

# Update pip in base and ptca environments
RUN conda install pip -n base -y
RUN conda install pip -n ptca -y

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.10 \
    pip \
    numpy~=1.23.5\
    libffi=3.4.4 \
    pycocotools=2.0.4 \
    shap=0.39.0 \
    llvmlite=0.39.1 \
    scipy=1.10.1 \
    setuptools=82.0.1 \
    wheel=0.46.2 \
    tbb=2021.1.1 \
    -c conda-forge -c cerebis && \
    conda clean -a -y

# Install pip dependencies
RUN pip install --no-cache-dir \
                azureml-mlflow==1.62.0.post5 \
                azureml-dataset-runtime==1.62.0 \
                azureml-telemetry==1.62.0 \
                azureml-responsibleai==1.62.0 \
                azureml-automl-core==1.62.0.post3 \
                azureml-automl-runtime==1.62.0.post1 \
                azureml-defaults==1.62.0 \
                azureml-interpret==1.62.0 \
                'azureml-dataprep>=2.24.4' \
                'azure-identity>=1.25.1' \
                'pillow==12.2.0'

# Install packages with torch packages separately to reduce layer size
RUN pip install --no-cache-dir \
                azureml-train-automl-client==1.62.0 \
                azureml-train-automl-runtime==1.62.0 \
                azureml-automl-dnn-vision==1.62.0

# Vulnerability patches for conda environment
# NOTE: cryptography>=48.0.1 (GHSA-537c-gmf6-5ccf): azureml-mlflow==1.62.0.post2
#       constrains cryptography<47.0.0, but no AzureML parent has a patched
#       cryptography floor; explicit override is required.
# NOTE: azureml-automl-runtime pins onnx<=1.17.0,>=1.16.1; force-installing onnx>=1.21.0 to fix
#       GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6, GHSA-p433-9wv8-28xj,
#       GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m. Parent azureml-automl-runtime (1.62.0) cannot
#       be upgraded as latest still constrains onnx<=1.17.0.
# NOTE: starlette>=1.0.1 (GHSA-86qp-5c8j-p5mr): starlette 0.52.1 is installed as a
#       transitive dep into the azureml-automl-dnn-vision-gpu conda env. No parent
#       release pins the patched floor; explicit override required.
# NOTE: pyarrow>=23.0.1 (GHSA-rgxp-2hwp-jwgg / CVE-2026-25087): pyarrow 17.0.0
#       is installed transitively via azureml-dataset-runtime in this conda env;
#       explicit override required until the AzureML parent pins a patched floor.
RUN pip install --no-cache-dir --upgrade \
                'cryptography>=48.0.1' \
                'distributed>=2026.1.0' \
                'mlflow-skinny>=2.16.0' \
                'bokeh>=3.8.2' \
                'pillow==12.2.0' \
                'onnx>=1.21.0' \
                'requests>=2.33.0' \
                'idna>=3.15' \
                'pyOpenSSL>=26.0.0' \
                'wheel>=0.46.2' \
                'starlette>=1.0.1' \
                'pyarrow>=23.0.1'

# Vulnerability patches for ptca environment
# pip>=26.1 (CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9): pip < 26.1 deferred imports of
#   well-known module names until after wheel install, allowing a freshly-installed
#   wheel to be imported during the self-update check. ptca env ships pip 26.0.1
#   in biweekly.202605.2; pip is its own parent (no upstream package can pull in a
#   fixed pip via dependency resolution), so explicit override is required.
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade \
                'pip>=26.1' \
                'aiohttp>=3.14.0'
# setuptools resolver picks wrong version due to dep conflicts; force install to fix jaraco.context vuln (GHSA-58pv-8j8x-9vj2)
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

# python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94 / CVE-2026-28684): set_key()/unset_key()
#   follow symlinks on cross-device .env writes, allowing arbitrary file overwrite.
#   python-dotenv 1.2.1 is shipped in the base conda env (python 3.13) of the ACPT
#   base image as a transitive dep via pydantic-settings (declares python-dotenv>=0.21.0
#   with no upper-bound tightening through 2.14.0), so no parent package release pins
#   python-dotenv>=1.2.2 — explicit override of /opt/conda is the only fix path until
#   the ACPT base image is rebuilt with the patch.
# pip>=26.1 (CVE-2026-6357 / GHSA-jp4c-xjxw-mgf9): pip < 26.1 deferred imports of
#   well-known module names until after wheel install, allowing a freshly-installed
#   wheel to be imported during the self-update check. Base conda env ships pip 26.0.1
#   from the ACPT base image; pip is its own parent (no upstream package can pull in
#   a fixed pip via dependency resolution), so explicit override is required.
# urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j): urllib3 2.6.3 is shipped
#   in the base conda env (py3.13) by the ACPT base image. urllib3 is a root security
#   package — its parents (requests pins urllib3>=1.21.1,<3; botocore pins
#   urllib3>=1.25.4,<3) do not bound it tightly enough to force 2.7.0, so no parent
#   upgrade can pull in the fix. Explicit override required.
# click>=8.3.3 (GHSA-47fr-3ffg-hgmw / CVE-2026-7246): metadata probe of the
#   resolved base image (biweekly.202605.2) found click 8.2.1 in /opt/conda via
#   anaconda-cli-base==0.8.2 (`click`) and typer==0.25.1 (`click>=8.2.1`).
#   Current published parent metadata still has loose floors only, so no parent
#   upgrade can force the patched click version; explicit override is required.
# aiohttp>=3.14.0 (GHSA-hg6j-4rv6-33pg / GHSA-jg22-mg44-37j8): aiohttp 3.13.5 is
#   shipped in the base conda env (py3.13) of the ACPT base image. Two CVEs require
#   3.14.0; no parent release pins the patched floor, so explicit override required.
# PyJWT>=2.13.0 (GHSA-993g-76c3-p5m4 / GHSA-jq35-7prp-9v3f): PyJWT 2.12.1 is
#   shipped in the base conda env (py3.13) of the ACPT base image; explicit
#   override required until the base image is rebuilt with a patched floor.
# py-rattler>=0.24.0 (GHSA-q53q-5r4j-5729): py-rattler 0.23.1 is shipped in the
#   base conda env (py3.13) of the ACPT base image; explicit override required
#   until the base image is rebuilt with a patched floor.
# cryptography>=48.0.1 (GHSA-537c-gmf6-5ccf): cryptography 46.0.7 is shipped in
#   the base conda env (py3.13) of the ACPT base image; explicit override required
#   until the base image is rebuilt with a patched floor.
RUN conda run -n base pip install --no-cache-dir --upgrade \
                'python-dotenv>=1.2.2' \
                'pip>=26.1' \
                'urllib3>=2.7.0' \
                'idna>=3.15' \
                'click>=8.3.3' \
                'aiohttp>=3.14.0' \
                'PyJWT>=2.13.0' \
                'py-rattler>=0.24.0' \
                'cryptography>=48.0.1'
# Fix vendored jaraco.context (GHSA-58pv-8j8x-9vj2) and wheel (GHSA-8rrh-rw8j-w5fx) in base setuptools
# setuptools vendors jaraco.context internally; --force-reinstall --no-deps ensures vendored copies are replaced
RUN /opt/conda/bin/pip install --no-cache-dir --force-reinstall --no-deps 'setuptools==82.0.1'

RUN rm -rf /opt/conda/pkgs/

# Remove stale conda-meta entries for pip 26.0.x in both base and ptca envs.
# `pip install --upgrade pip` (above) replaces the wheel under site-packages but
# does NOT update conda's package database (conda-meta/pip-*.json), so SCA
# scanners that read conda-meta keep flagging CVE-2026-6357 even though the
# running pip is 26.1.x. Removing the stale json makes scanners pick up the
# pip dist-info from site-packages (which reflects the upgraded version).
RUN rm -f /opt/conda/conda-meta/pip-26.0.*.json \
          /opt/conda/envs/ptca/conda-meta/pip-26.0.*.json

ENV LD_LIBRARY_PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
