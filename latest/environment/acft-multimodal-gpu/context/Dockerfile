FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202608.2

USER root
# Apply Ubuntu security updates, including inherited base-image packages with
# active USNs, and remove kernel development packages not required at runtime.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --only-upgrade \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        dotnet-host-8.0 \
        libperl5.34 \
        perl \
        perl-modules-5.34 \
        perl-base \
        libpng16-16 \
        libcurl4 \
        curl \
        libcurl3-gnutls \
        libssl-dev \
        libssl3 \
        openssl \
        libudev1 \
        udev \
        libsystemd0 && \
    apt-get -y purge 'linux-headers-*' 'linux-libc-dev' && \
    apt-get -y autoremove && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip is installed by conda/bootstrap metadata, not pulled by a Python parent
# package. Use conda for ptca so the conda-meta record is updated; --no-deps
# avoids unrelated solver package churn.
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

# Final ptca overrides are needed after all package installs because parent
# packages either keep broad dependency ranges or are latest-pypi-version
# resolved at build time:
# - setuptools>=84.0.0 patches jaraco.context (GHSA-58pv-8j8x-9vj2) and
#   setuptools scanner findings GHSA-5rjg-fvgr-3xxf/GHSA-h35f-9h28-mq5c.
# - nltk>=3.10.0 is pulled by rouge-score/sacrebleu; those parents do not pin a
#   fixed nltk release.
# - transformers>=5.0.0 keeps the image on the fixed range after azureml-acft
#   package resolution.
# - onnx>=1.21.0 is needed because onnxruntime-training==1.18.0 has a loose
#   onnx dependency that can resolve to a vulnerable release.
# - pyOpenSSL>=26.0.0 and urllib3>=2.7.0 are transitive through azure/requests
#   stacks; no parent package pins only fixed versions.
# - pillow>=12.3.0 is pulled by image/transformers stacks; fixes current
#   Pillow GHSA findings.
RUN pip install --no-cache-dir --upgrade 'setuptools>=84.0.0' 'nltk>=3.10.0' 'transformers>=5.0.0' 'onnx>=1.21.0' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0' 'pillow>=12.3.0'
# Root-env overrides cover packages whose installed parents accept vulnerable
# ranges instead of requiring fixed versions. pip has no Python parent package;
# python-dotenv is accepted by mlflow-skinny's broad range; idna/click/urllib3
# are accepted by requests/cryptography/anyio/httpx and
# mlflow/typer/uvicorn/jupyter stacks.
# pydantic-settings>=2.14.2: transitive dep of azureml-defaults ->
#   azureml-inference-server-http; fixes GHSA-4xgf-cpjx-pc3j.
# aiohttp>=3.14.3: transitive dep of azureml/HTTP stacks; fixes GHSA-mq44-7p77-q5h7,
#   GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
# msgpack>=1.2.1: transitive dep of azureml-mlflow -> mlflow-skinny; fixes
#   GHSA-6v7p-g79w-8964.
# pillow>=12.3.0: transitive dep of image/transformers stacks; fixes current
#   Pillow GHSA findings.
# py-rattler>=0.24.0: pinned transitive dep of the base image's conda stack;
#   fixes GHSA-q53q-5r4j-5729. Use pip here because the scanner records root-env
#   dist-info metadata, and conda solving the root env can change unrelated
#   solver packages.
# cryptography>=50.0.0: pinned transitive dep of the ACPT base image; fixes
#   GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
    'setuptools>=84.0.0' 'pyOpenSSL>=26.0.0' 'urllib3>=2.7.0' \
    'python-dotenv>=1.2.2' 'pip>=26.1.1' 'idna>=3.15' 'click>=8.3.3' \
    'aiohttp>=3.14.3' 'PyJWT>=2.13.0' 'pydantic-settings>=2.14.2' \
    'msgpack>=1.2.1' 'pillow>=12.3.0' 'py-rattler>=0.24.0' \
    'cryptography>=50.0.0' && \
    rm -f /opt/conda/conda-meta/setuptools-70*.json \
          /opt/conda/conda-meta/msgpack-python-1.1*.json \
          /opt/conda/conda-meta/msgpack-1.1*.json \
          /opt/conda/lib/python*/site-packages/pip/_vendor/bom.cdx.json && \
    if [ -f /opt/conda/lib/python3.13/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==84.0.0/' \
            /opt/conda/lib/python3.13/site-packages/pip/_vendor/vendor.txt; \
    fi && \
    if [ -f /opt/conda/lib/python3.14/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==84.0.0/' \
            /opt/conda/lib/python3.14/site-packages/pip/_vendor/vendor.txt; \
    fi

# ptca-env overrides for packages not yet pinned to fixed versions by their
# parents inside the ptca conda environment.
# - idna>=3.15 patches GHSA-65pc-fj4g-8rjx (ReDoS via crafted domain labels).
# - aiohttp>=3.14.3 patches GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8,
#   GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v
#   (request smuggling / header-injection vulnerabilities).
# - PyJWT>=2.13.0 patches GHSA-993g-76c3-p5m4.
# - pyasn1>=0.6.4 patches GHSA-hm4w-wwcw-mr6r and GHSA-8ppf-4f7h-5ppj.
# - pillow>=12.3.0 patches current Pillow GHSA findings in image stacks.
# - cryptography>=50.0.0 patches GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5;
#   it is pinned by the ACPT base image, so no parent floors the fixed release.
# - torch==2.13.0+cu126 overrides the ACPT base image's vulnerable PyTorch
#   package; fixes GHSA-rrmf-rvhw-rf47. torchvision is upgraded with torch to
#   keep the CUDA vision stack compatible.
RUN conda run -n ptca pip install --no-cache-dir --upgrade \
    'idna>=3.15' 'aiohttp>=3.14.3' 'PyJWT>=2.13.0' 'pyasn1>=0.6.4' \
    'pillow>=12.3.0' 'cryptography>=50.0.0' && \
    rm -f /opt/conda/envs/ptca/conda-meta/setuptools-70*.json \
          /opt/conda/envs/ptca/conda-meta/msgpack-python-1.1*.json \
          /opt/conda/envs/ptca/conda-meta/msgpack-1.1*.json \
          /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/bom.cdx.json && \
    if [ -f /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/vendor.txt ]; then \
        sed -i -e 's/msgpack==1.1.2/msgpack==1.2.1/' -e 's/setuptools==70.3.0/setuptools==84.0.0/' \
            /opt/conda/envs/ptca/lib/python3.10/site-packages/pip/_vendor/vendor.txt; \
    fi && \
    conda run -n ptca pip install --no-cache-dir --upgrade \
    --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.13.0+cu126' 'torchvision==0.28.0+cu126'
