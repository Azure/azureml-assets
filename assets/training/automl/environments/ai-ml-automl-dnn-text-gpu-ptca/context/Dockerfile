FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root:root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

RUN pip install --no-cache-dir \
    'azureml-automl-dnn-nlp=={{latest-pypi-version}}' \
    'azureml-defaults=={{latest-pypi-version}}'

# onnx and onnxruntime-training installation
RUN pip uninstall -y onnxruntime
RUN pip uninstall -y onnxruntime-training
RUN pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-training==1.18.0

# torch-ort installation
RUN TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
RUN pip install torch-ort==1.18.0 && TORCH_CUDA_ARCH_LIST="5.2;6.0;7.0;8.0;8.6;9.0" python -m torch_ort.configure
RUN pip uninstall -y onnxruntime

RUN pip install \
    optimum==1.23.3 \
    accelerate==1.12.0 \
    deepspeed~=0.15.1

# Override transformers to fix GHSA-69w3-r845-3855
# Root cause: azureml-automl-dnn-nlp==1.62.0 is the latest release as of
# 2026-05-26 and pins transformers==4.53.0, so direct override is required.
RUN pip install --no-cache-dir --no-deps 'transformers[sentencepiece,torch]==5.5.4'

# Vulnerability patches for ptca environment (python 3.10 at /opt/conda/envs/ptca)
# pip 26.0.1 -> >=26.1.1 (GHSA-jp4c-xjxw-mgf9, CVE-2026-6357): no parent
#   package controls pip here, so direct upgrade is required. The stale
#   conda-meta JSON for pip-26.0.1 is removed so scanners do not re-flag it.
# setuptools 81.0.0 -> >=82.0.1 (GHSA-58pv-8j8x-9vj2 in vendored jaraco.context):
#   setuptools is a build dependency with no parent that pins it.
# Override onnx to fix GHSA-cmw6-hcpp-c6jp, GHSA-538c-55jv-c5g9, GHSA-q56x-g2fj-4rj6,
#   GHSA-p433-9wv8-28xj, GHSA-3r9x-f23j-gc73, GHSA-hqmj-h5c6-369m.
#   Root cause: azureml-automl-runtime==1.62.0 is the latest release as of
#   2026-05-26 and pins onnx<=1.17.0, so direct override is required.
# bokeh 2.4.3 -> >=3.8.2 (GHSA-793v-589g-574v, CVE-2026-21883) and distributed
#   2023.2.0 -> >=2026.1.0 (GHSA-c336-7962-wfj2, CVE-2026-23528) are pulled in by
#   azureml-train-automl-runtime==1.62.0, which requires bokeh<3.0.0 and
#   dask[complete]<=2023.2.0. Latest AzureML parents remain at 1.62.0 as of
#   2026-05-26, so direct override is required.
# aiohttp 3.13.5 -> >=3.14.0 (GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg): aiohttp
#   is pulled in transitively with no parent pinning it to <3.14.0, so direct
#   override is required.
# pyarrow 17.0.0 -> >=23.0.1 (GHSA-rgxp-2hwp-jwgg / CVE-2026-25087): pinned
#   transitive dep of azureml-automl-dnn-nlp via AzureML dataset/runtime packages.
# cryptography 46.0.7 -> >=48.0.1 (GHSA-537c-gmf6-5ccf): pinned transitive dep
#   brought in by the ACPT base image ptca conda environment.
RUN /opt/conda/envs/ptca/bin/pip install --no-cache-dir --upgrade \
    'pip>=26.1.1' \
    'setuptools>=82.0.1' \
    'onnx>=1.21.0' \
    'bokeh>=3.8.2' \
    'distributed>=2026.1.0' \
    'aiohttp>=3.14.0' \
    'pyarrow>=23.0.1' \
    'cryptography>=48.0.1' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0.1-*.json

# Vulnerability patches for conda base env (python 3.13 at /opt/conda)
# pip 26.0.1 -> >=26.1.1 (GHSA-jp4c-xjxw-mgf9, CVE-2026-6357): no parent
#   package controls pip here, so direct upgrade is required.
# urllib3 2.6.3 -> >=2.7.0 (GHSA-qccp-gfcp-xxvc / CVE-2026-44431, GHSA-mf9v-mfxr-j63j
#   / CVE-2026-44432): requests 2.34.2 is the latest release as of 2026-05-26 and
#   still allows urllib3<3,>=1.26; conda CLI parents also use loose urllib3 ranges,
#   so no parent upgrade can force >=2.7.0 and direct override is required.
# idna >=3.15 (GHSA-65pc-fj4g-8rjx): parents (`requests`, `anyio`, `httpx`,
#   `yarl`) declare loose idna requirements, so direct override is required.
# click >=8.3.3 (GHSA-47fr-3ffg-hgmw): CLI parents use loose click floors, so
#   direct override is required.
# setuptools 82.0.0 -> >=82.0.1 (GHSA-58pv-8j8x-9vj2 in vendored jaraco.context):
#   no parent package pins setuptools.
# python-dotenv 1.2.1 -> >=1.2.2 (GHSA-mf9w-mj56-hr94): brought in transitively by
#   anaconda-auth==0.14.4 (Requires-Dist: python-dotenv with no version pin) and
#   pydantic-settings==2.12.0 (python-dotenv>=0.21.0, via anaconda-cli-base ->
#   anaconda-auth). Latest releases on PyPI as of 2026-05-26 (anaconda-auth==0.15.0,
#   pydantic-settings==2.14.1) still use the same loose floors, so a parent upgrade
#   cannot force >=1.2.2 -- direct override required.
# aiohttp 3.13.5 -> >=3.14.0 (GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg): aiohttp
#   is pulled in transitively with no parent pinning it to <3.14.0, so direct
#   override is required.
# PyJWT 2.12.1 -> >=2.13.0 (GHSA-jq35-7prp-9v3f, GHSA-993g-76c3-p5m4): pinned
#   transitive dep brought in by the ACPT base image conda environment.
# py-rattler 0.23.1 -> >=0.24.0 (GHSA-q53q-5r4j-5729): pinned transitive dep
#   brought in by the ACPT base image conda tooling.
# cryptography 46.0.7 -> >=48.0.1 (GHSA-537c-gmf6-5ccf): pinned transitive dep
#   brought in by the ACPT base image conda environment.
# pydantic-settings 2.12.0 -> >=2.14.2 (GHSA-4xgf-cpjx-pc3j): pinned transitive
#   dep of anaconda-auth/anaconda-cli-base brought in by the ACPT base image conda tooling.
# Stale conda pkgs-cache entries are removed so scanners do not re-flag old versions
#   from the package cache directories.
RUN /opt/conda/bin/pip install --no-cache-dir --upgrade \
    'pip>=26.1.1' \
    'urllib3>=2.7.0' \
    'idna>=3.15' \
    'click>=8.3.3' \
    'setuptools>=82.0.1' \
    'python-dotenv>=1.2.2' \
    'aiohttp>=3.14.0' \
    'PyJWT>=2.13.0' \
    'py-rattler>=0.24.0' \
    'cryptography>=48.0.1' \
    'pydantic-settings>=2.14.2' && \
    rm -rf /opt/conda/pkgs/click-8.2.1-* \
           /opt/conda/pkgs/idna-3.11-* \
           /opt/conda/pkgs/pip-26.0.1-* \
           /opt/conda/pkgs/python-dotenv-1.2.1-* \
           /opt/conda/pkgs/pydantic-settings-2.12.0-*
