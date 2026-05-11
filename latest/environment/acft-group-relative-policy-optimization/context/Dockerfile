#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

USER root
# Security: upgrade all OS packages to patch USN vulnerabilities (libssh, curl, openssl, etc.)
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow==0.0.89
RUN pip install azureml-acft-common-components==0.0.89
# transformers 5.5.4: fixes GHSA-69w3-r845-3855 (arbitrary code execution in Trainer class);
# advisory requires >=5.0.0rc3; upgrading to latest stable 5.x
RUN pip install transformers==5.5.4

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635); upgrade after azureml packages
# azureml-evaluate-mlflow → azureml-mlflow pins mlflow-skinny<=3.9.0, conflicting with mlflow 3.10.1
# so mlflow must be upgraded separately to avoid pip resolution conflict
RUN pip install --no-cache-dir mlflow==3.10.1

# Upgrade transitive deps to fix vulnerabilities.
# Packages already at safe versions in base image biweekly.202605.1 (pip, wheel, setuptools,
# aiohttp, requests, urllib3, onnx, pillow, filelock, pytest) are NOT overridden here.
# protobuf: mlflow-skinny requires <7; base has 7.34.1 so we must pin <7 (GHSA fix >=6.33.5)
# cryptography: not in ptca env; installed transitively by azureml-mlflow (<47.0.0 constraint)
# ray: GHSA-q5fh-2hc8-f6rq; also bundles log4j-core 2.25.3 in JARs (patched below)
# azure-core, cbor2, jaraco.context, PyJWT: transitive deps with loose floors
# nltk: GHSA-gfwx-w7gr-fvh7; pyasn1/python-multipart/xgrammar: CVE floors
# mlflow: azureml-mlflow pins mlflow-skinny<=3.9.0; override needed
# vllm: GHSA-pq5c-rjhq-qp7p, GHSA-3mwp-wvh9-7528, GHSA-6c4r-fmh3-7rh8 (librosa LFE); >=0.20.1 required
#   vllm is a top-level dep in requirements.txt; pin floor here as a safety net
# pyOpenSSL: GHSA-vp96-hxj8-p424, GHSA-5pwr-322w-8jr4; >=26.0.0 required
#   azureml-core 1.61.0.post3 pins pyopenssl<26.0.0; no newer azureml-core available (2026-05-04)
RUN pip install --upgrade 'protobuf>=6.33.5,<7' 'cryptography>=46.0.7,<47' 'xgrammar>=0.1.32' \
    'nltk>=3.9.4' 'pyasn1>=0.6.3' 'python-multipart>=0.0.22' 'ray>=2.55.0' \
    'azure-core>=1.38.0' 'cbor2>=5.9.0' 'jaraco.context>=6.1.0' 'PyJWT>=2.12.0' \
    'mlflow>=3.8.1,<4.0.0' 'vllm>=0.20.1' 'pyOpenSSL>=26.0.0'
RUN rm -rf ~/.cache/pip
# python-dotenv in base conda env: GHSA-mf9w-mj56-hr94; base ships 1.2.1, need >=1.2.2
#   parent packages: uvicorn (>=0.13), pydantic-settings (>=0.21.0) — both use loose floors
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
# ray vendors aiohttp for runtime_env agent under thirdparty_files; patch that copy too.
RUN rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/__private/runtime_env/agent/thirdparty_files/aiohttp* && \
    pip install --no-cache-dir --target /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/__private/runtime_env/agent/thirdparty_files 'aiohttp==3.13.5'
# Patch log4j-core 2.25.3→2.25.4 inside ray's fat JAR (ray_dist.jar).
# ray (all versions through 2.55.1) bundles log4j-core 2.25.3 in java/dependencies.bzl.
# No ray release includes 2.25.4; manual JAR surgery is required.
# Fixes: GHSA-3pxv-7cmr-fjr4, GHSA-445c-vh5m-36rj, GHSA-6hg6-v5c8-fphq
COPY patch_log4j /tmp/patch_log4j.py
RUN python3 /tmp/patch_log4j.py 2.25.4 && rm -f /tmp/patch_log4j.py
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
