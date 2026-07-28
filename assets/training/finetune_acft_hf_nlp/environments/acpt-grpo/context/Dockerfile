FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
ENV PIP_PROGRESS_BAR=off
# Pull in latest OS security updates on top of the base image, including explicit
# floors for wget/tar/Kerberos/SQLite packages inherited from the base image.
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get -y -qq upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --only-upgrade \
        wget=1.21.2-2ubuntu1.4 \
        tar=1.34+dfsg-1ubuntu0.1.22.04.6 \
        libkrb5support0=1.19.2-2ubuntu0.8 \
        libgssapi-krb5-2=1.19.2-2ubuntu0.8 \
        libk5crypto3=1.19.2-2ubuntu0.8 \
        libkrb5-3=1.19.2-2ubuntu0.8 \
        libsqlite3-0=3.37.2-2ubuntu0.7 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip install azureml-evaluate-mlflow=={{latest-pypi-version}}
RUN pip install azureml-acft-common-components=={{latest-pypi-version}}

# Security floors for transitive deps whose parent packages do not pin them safely.
# protobuf:        mlflow-skinny<=3.9.0 requires protobuf<7 (GHSA fix at >=6.33.5)
# ray:             GHSA-q5fh-2hc8-f6rq; also bundles log4j-core 2.25.3 in JARs (patched below)
# nltk:            GHSA-gfwx-w7gr-fvh7
# pyasn1, python-multipart, xgrammar: CVE floors (no parent that pins them safely)
# mlflow:          azureml-mlflow 1.62.0.post2 pins mlflow-skinny<=3.9.0; explicit upgrade
#                  required to clear CVE-2025-14287, CVE-2026-2033, CVE-2026-2635
# vllm:            GHSA-pq5c-rjhq-qp7p, GHSA-3mwp-wvh9-7528, GHSA-6c4r-fmh3-7rh8 (librosa LFE)
# pyOpenSSL:       GHSA-vp96-hxj8-p424, GHSA-5pwr-322w-8jr4; azureml-core 1.61.0.post3 pins
#                  pyopenssl<26.0.0, so explicit override is the only remediation.
# cryptography:    pinned transitive/base image package in ptca env; fixes GHSA-537c-gmf6-5ccf.
# urllib3:         GHSA-mf9v-mfxr-j63j, GHSA-qccp-gfcp-xxvc; fix in 2.7.0. urllib3 is brought
#                  in transitively by requests/botocore/azureml-core/kubernetes/etc., all of
#                  which only constrain urllib3<3 (loose). No parent package pins urllib3>=2.7,
#                  so direct override is the only remediation.
# vllm:            GHSA-rwxx-mrjm-wc2m and related vLLM advisories require >=0.24.0.
RUN pip install --upgrade 'protobuf>=6.33.5,<7' 'xgrammar>=0.1.32' \
    'nltk>=3.9.4' 'pyasn1>=0.6.3' 'python-multipart>=0.0.22' 'ray>=2.55.0' \
    'mlflow>=3.8.1,<4.0.0' 'vllm>=0.24.0' 'pyOpenSSL>=26.0.0' 'cryptography>=48.0.1' \
    'urllib3>=2.7.0'
# torch: pinned transitive dep of vllm; vllm 0.24.0-0.26.0 pins torch==2.11.0,
# so install the GHSA-rrmf-rvhw-rf47 security floor separately. transformers:
# vllm requires transformers<5, so reinstall the scanner-required 5.x floor after vllm.
RUN pip install --no-cache-dir --upgrade --no-deps 'torch>=2.13.0' 'transformers==5.5.4'
# aiohttp: GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8 (fix in >=3.14.0)
# starlette: GHSA-86qp-5c8j-p5mr (fix in >=1.0.1)
# idna: GHSA-65pc-fj4g-8rjx (fix in >=3.15)
RUN pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0' 'starlette>=1.0.1' 'idna>=3.15'
# pip: GHSA-wf93-45jw-7689; fixed in 26.1.2. pip is the Python
# package installer itself - it has NO parent/transitive package that pulls it in
# (it is bootstrapped by the conda/python distribution), so the only available
# remediation is to upgrade pip directly in each conda environment.
# Also remove stale conda metadata because `pip install --upgrade pip` does not
# update conda's metadata DB, and SBOM scanners read those records.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1.2' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-26.0*.json \
          /opt/conda/envs/ptca/conda-meta/pip-26.1.1-*.json
RUN rm -rf ~/.cache/pip
# python-dotenv in base conda env: GHSA-mf9w-mj56-hr94; base ships 1.2.1, need >=1.2.2
#   parent packages: uvicorn (>=0.13), pydantic-settings (>=0.21.0) - both use loose floors
# urllib3 in base conda env: see urllib3 note above; same direct-override rationale.
# pip: see note above - bootstrap tool, must be upgraded directly in base env too.
# py-rattler: pinned transitive dep from base image conda tooling; fixes GHSA-q53q-5r4j-5729
# cryptography: pinned transitive/base image package in base env; fixes GHSA-537c-gmf6-5ccf.
# pydantic-settings: pinned transitive dep from base image conda tooling; fixes GHSA-4xgf-cpjx-pc3j
# Also clean stale conda-meta/pip-*.json (see ptca env note above for scanner detail).
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'pip>=26.1.2' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' \
        'aiohttp>=3.14.0' 'click>=8.3.3' 'idna>=3.15' \
        'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=48.0.1' 'pydantic-settings>=2.14.2' && \
        rm -f /opt/conda/conda-meta/pip-26.0*.json \
              /opt/conda/conda-meta/pip-26.1.1-*.json
# pyarrow: GHSA-rgxp-2hwp-jwgg (fix in >=23.0.1); ptca env ships 20.0.0 via vllm transitive dep.
# PyJWT: GHSA-jq35-7prp-9v3f, GHSA-993g-76c3-p5m4 (fix in >=2.13.0); base env ships 2.12.1.
RUN pip install --no-cache-dir --upgrade 'pyarrow>=23.0.1'
# ray vendors aiohttp/idna under _private/runtime_env/agent/thirdparty_files (single underscore).
# The actual Python module path is ray/_private (not ray/__private).
RUN rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/_private/runtime_env/agent/thirdparty_files/aiohttp* \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/_private/runtime_env/agent/thirdparty_files/idna* && \
    pip install --no-cache-dir --target /opt/conda/envs/ptca/lib/python3.10/site-packages/ray/_private/runtime_env/agent/thirdparty_files 'aiohttp>=3.14.0' 'idna>=3.15'
# Patch log4j-core 2.25.3 to 2.25.4 inside Ray's fat JAR.
# ray (all versions through 2.55.1) bundles log4j-core 2.25.3 in java/dependencies.bzl.
# No ray release includes 2.25.4; manual JAR surgery is required.
# Fixes: GHSA-3pxv-7cmr-fjr4, GHSA-445c-vh5m-36rj, GHSA-6hg6-v5c8-fphq
COPY patch_log4j /tmp/patch_log4j.py
RUN python3 /tmp/patch_log4j.py 2.25.4 && rm -f /tmp/patch_log4j.py
# jackson-core/databind: pinned transitive deps of ray; bundled in Ray's fat JAR,
# fixes GHSA-rmj7-2vxq-3g9f, GHSA-j3rv-43j4-c7qm, GHSA-hgj6-7826-r7m5,
# GHSA-5jmj-h7xm-6q6v, GHSA-5gvw-p9qm-jgwh, GHSA-mhm7-754m-9p8w,
# and GHSA-r7wm-3cxj-wff9.
COPY patch_ray_jackson_databind.py /tmp/patch_ray_jackson_databind.py
RUN JACKSON_ARTIFACT=jackson-core \
    JACKSON_VERSION=2.18.8 \
    JACKSON_JAR_SHA1=459eb2c4bb30bd6898eb9bbe9d7a4c4140529b8b \
    JACKSON_POM_SHA1=90813471a8fd344b176d7459ab2ee310dacb607f \
    python3 /tmp/patch_ray_jackson_databind.py && \
    JACKSON_ARTIFACT=jackson-databind \
    JACKSON_VERSION=2.18.9 \
    JACKSON_JAR_SHA1=093f3c1564ae82c1f5c034883f9ce45782054e04 \
    JACKSON_POM_SHA1=e2f1e832d86886fc316342c59503090191645fb9 \
    python3 /tmp/patch_ray_jackson_databind.py && \
    rm -f /tmp/patch_ray_jackson_databind.py
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
