#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202605.1

USER root
# Pull in latest OS security updates on top of the base image
RUN apt-get update && apt-get -y upgrade && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow==0.0.90
RUN pip install azureml-acft-common-components==0.0.90
# transformers 5.5.4: fixes GHSA-69w3-r845-3855 (arbitrary code execution in Trainer class);
# advisory requires >=5.0.0rc3; upgrading to latest stable 5.x
RUN pip install transformers==5.5.4

# Security floors for transitive deps that the base image / parent packages do not pin tightly.
# Removed pins (now redundant against base image biweekly.202605.1 + parent constraints):
#   cryptography  - azureml-mlflow already constrains <47.0.0 -> pip resolves to 46.0.7 (safe)
#   azure-core    - azureml-mlflow constrains >=1.8.0,<2.0.0  -> pip resolves to latest 1.41.x
#   PyJWT         - azureml-core constrains <3.0.0 and base ships 2.12.1 (>= floor)
#   jaraco.context - base env ships 6.1.0 (= floor); ptca env resolves naturally to 6.1.x
#   cbor2         - vllm depends on cbor2 unconstrained; pip picks current latest 6.x
# Remaining floors below are still required because the parent's resolved version is at or near
# the vulnerability boundary, or the parent package itself ships a vulnerable version:
# protobuf:        mlflow-skinny requires <7; base ships 7.34.1 so pin <7 (GHSA fix >=6.33.5)
# ray:             GHSA-q5fh-2hc8-f6rq; also bundles log4j-core 2.25.3 in JARs (patched below)
# nltk:            GHSA-gfwx-w7gr-fvh7
# pyasn1, python-multipart, xgrammar: CVE floors (no parent that pins them safely)
# mlflow:          azureml-mlflow pins mlflow-skinny<=3.9.0; explicit upgrade required to clear
#                  CVE-2025-14287, CVE-2026-2033, CVE-2026-2635
# vllm:            GHSA-pq5c-rjhq-qp7p, GHSA-3mwp-wvh9-7528, GHSA-6c4r-fmh3-7rh8 (librosa LFE)
# pyOpenSSL:       GHSA-vp96-hxj8-p424, GHSA-5pwr-322w-8jr4; azureml-core 1.61.0.post3 pins
#                  pyopenssl<26.0.0 and no newer azureml-core release exists (verified 2026-05-14),
#                  so explicit override is the only remediation.
RUN pip install --upgrade 'protobuf>=6.33.5,<7' 'xgrammar>=0.1.32' \
    'nltk>=3.9.4' 'pyasn1>=0.6.3' 'python-multipart>=0.0.22' 'ray>=2.55.0' \
    'mlflow>=3.8.1,<4.0.0' 'vllm>=0.20.1' 'pyOpenSSL>=26.0.0'
# pip: GHSA-jp4c-xjxw-mgf9 (CVE-2026-6357); base image biweekly.202605.1 ships 26.0.1 in
# both ptca (py3.10) and base (py3.13) conda envs, fixed in 26.1. pip is the Python
# package installer itself — it has NO parent/transitive package that pulls it in
# (it is bootstrapped by the conda/python distribution), so the only available
# remediation is to upgrade pip directly in each conda environment.
# Also remove stale conda-meta/pip-*.json from ptca env: `pip install --upgrade pip`
# replaces the on-disk pip files but does NOT update conda's metadata DB, so SBOM
# scanners that read conda-meta still report the old pip version.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1.1' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json
RUN rm -rf ~/.cache/pip
# python-dotenv in base conda env: GHSA-mf9w-mj56-hr94; base ships 1.2.1, need >=1.2.2
#   parent packages: uvicorn (>=0.13), pydantic-settings (>=0.21.0) — both use loose floors
# pip: see note above — bootstrap tool, must be upgraded directly in base env too
# Also clean stale conda-meta/pip-*.json (see ptca env note above for scanner detail).
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'pip>=26.1.1' 'python-dotenv>=1.2.2' && \
    rm -f /opt/conda/conda-meta/pip-26.0*.json
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
