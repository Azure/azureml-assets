# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.2

USER root
# Apply Ubuntu security updates and remove kernel development packages that are
# not required at container runtime.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y purge 'linux-headers-*' 'linux-libc-dev' && \
    apt-get -y autoremove && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Checked 2026-05-25: pip is installed by conda/bootstrap metadata, not pulled
# by a Python parent package. Use conda for ptca so the conda-meta record is
# updated; --no-deps avoids unrelated solver package churn.
RUN conda install -n ptca -c conda-forge -y --no-deps "pip>=26.1.1" && \
    conda clean -afy

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# Downgraded azureml-metrics, to prevent bug related to logger in azureml-metrics. Should be upgraded in next pypi release of acft-multimodal
# Removing it from requirements.txt to avoid conflict with scikit-learn version
RUN pip install azureml-metrics==0.0.33 pyarrow==14.0.1

RUN pip install azureml-acft-common-components==0.0.91
RUN pip install azureml-acft-accelerator==0.0.91

# Checked 2026-05-25: final ptca overrides are still needed after all package
# installs because parent packages either keep broad dependency ranges or are
# latest-pypi-version resolved at build time:
# - setuptools>=82.0.1 patches jaraco.context (GHSA-58pv-8j8x-9vj2).
# - nltk>=3.9.4 is pulled by rouge-score/sacrebleu; those parents do not pin a
#   fixed nltk release.
# - transformers>=5.0.0 keeps the image on the fixed range after azureml-acft
#   package resolution.
# - onnx>=1.21.0 is needed because onnxruntime-training==1.18.0 has a loose
#   onnx dependency that can resolve to a vulnerable release.
# - pyOpenSSL>=26.0.0 and urllib3>=2.7.0 are transitive through azure/requests
#   stacks; no parent package pins only fixed versions.
RUN pip install --no-cache-dir --upgrade 'setuptools>=82.0.1' 'nltk>=3.9.4' 'transformers>=5.0.0' 'onnx>=1.21.0' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0'
# Checked 2026-05-25: root-env overrides cover packages whose installed parents
# accept vulnerable ranges instead of requiring fixed versions. pip has no
# Python parent package; python-dotenv is accepted by mlflow-skinny's broad
# range; idna/click/urllib3 are accepted by requests/cryptography/anyio/httpx
# and mlflow/typer/uvicorn/jupyter stacks. Use pip here because the scanner
# records root-env dist-info metadata, and conda solving the root env can change
# unrelated solver packages.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
    'setuptools>=82.0.1' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0' \
    'python-dotenv>=1.2.2' 'pip>=26.1.1' 'idna>=3.15' 'click>=8.3.3' \
    'aiohttp>=3.14.0' 'PyJWT>=2.13.0'

# Checked 2026-06-16: ptca-env overrides for packages not yet pinned to fixed
# versions by their parents inside the ptca conda environment.
# - idna>=3.15 patches GHSA-65pc-fj4g-8rjx (ReDoS via crafted domain labels).
# - aiohttp>=3.14.0 patches GHSA-hg6j-4rv6-33pg and GHSA-jg22-mg44-37j8
#   (request smuggling / header-injection vulnerabilities).
# - PyJWT>=2.13.0 patches GHSA-993g-76c3-p5m4.
RUN conda run -n ptca pip install --no-cache-dir --upgrade \
    'idna>=3.15' 'aiohttp>=3.14.0' 'PyJWT>=2.13.0'
