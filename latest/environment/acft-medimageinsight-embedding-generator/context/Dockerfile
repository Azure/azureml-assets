FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202608.2

USER root

# Install unzip and upgrade OS packages to fix vulnerabilities.
# The glibc, krb5, PAM, ALSA, nghttp2 and dotnet entries below are inherited unpatched from
# the ACPT base image, whose tag lags the Ubuntu 22.04 security archive. Remove them from this
# --only-upgrade list once the base image ships the fixed versions on its own.
RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --only-upgrade \
        curl \
        dotnet-host-8.0 \
        dotnet-hostfxr-8.0 \
        dotnet-runtime-8.0 \
        gzip \
        libasound2 \
        libasound2-data \
        libc-bin \
        libc-dev-bin \
        libc6 \
        libc6-dev \
        libcurl3-gnutls \
        libcurl4 \
        libgssapi-krb5-2 \
        libk5crypto3 \
        libkrb5-3 \
        libkrb5support0 \
        libnghttp2-14 \
        libpam-modules \
        libpam-modules-bin \
        libpam-runtime \
        libpam0g \
        libpython3.10-minimal \
        libpython3.10-stdlib \
        libsqlite3-0 \
        locales \
        openssh-client \
        python3.10 \
        python3.10-minimal \
        tar \
        wget && \
    apt-get install -y --no-install-recommends unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the patched 26.1 line in both envs to fix GHSA-jp4c-xjxw-mgf9 (bootstrap
# installer, no parent to bump). Installed from PyPI; stale conda-meta JSON removed so the
# SBOM scanner doesn't re-flag it. Done before requirements so later installs use patched pip.
RUN pip install --no-cache-dir --upgrade 'pip>=26.1.2,<27' && \
    rm -f /opt/conda/envs/ptca/conda-meta/pip-*.json && \
    /opt/conda/bin/python -m pip install --no-cache-dir --upgrade 'pip>=26.1.2,<27' && \
    rm -f /opt/conda/conda-meta/pip-*.json

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# ptca-env (py3.10) transitive deps pip won't auto-upgrade; no parent floors the safe versions:
# Mako>=1.3.11 (GHSA-v92g-xgxw-vvmm), GitPython>=3.1.58 (GHSA-x2qx-6953-8485, GHSA-rpm5-65cw-6hj4,
# GHSA-2f96-g7mh-g2hx, GHSA-956x-8gvw-wg5v, GHSA-v396-v7q4-x2qj, GHSA-94p4-4cq8-9g67,
# GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, GHSA-r9mr-m37c-5fr3, GHSA-p538-c434-8v24,
# GHSA-539m-9xh6-q6rr, and GHSA-3f7w-8rr8-f37f),
# pytest>=9.0.3 (GHSA-6w46-j5rx-g56g), skops>=0.13.0 (CVE-2025-54412/54413/54886),
# urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j), aiohttp>=3.14.3 (GHSA-hg6j-4rv6-33pg,
# GHSA-jg22-mg44-37j8, GHSA-mfx4-hv73-q22v, GHSA-cq5v-8q36-5273, GHSA-mq44-7p77-q5h7),
# pyarrow>=23.0.1 (GHSA-rgxp-2hwp-jwgg).
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes
#   GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# pyasn1: transitive dep of mlflow -> databricks-sdk -> google-auth; no parent floors the safe
#   version, fixes GHSA-hm4w-wwcw-mr6r and GHSA-8ppf-4f7h-5ppj.
# pillow: transitive dep of the image/vision packages; fixes GHSA-8v84-f9pq-wr9x,
#   GHSA-9hw9-ch79-4vh6, GHSA-vjc4-5qp5-m44j, GHSA-pg7v-jwj7-p798, GHSA-45hq-cxwh-f6vc,
#   GHSA-phj9-mv4w-65pm, GHSA-62p4-gmf7-7g93, and GHSA-5x94-69rx-g8h2.
# msgpack: transitive dep of the AzureML Dask stack; fixes GHSA-6v7p-g79w-8964.
# setuptools: bootstrapped in the ptca env; fixes GHSA-h35f-9h28-mq5c.
RUN pip install --no-cache-dir --upgrade 'Mako>=1.3.11' 'GitPython>=3.1.58' 'pytest>=9.0.3' 'skops>=0.13.0' 'urllib3>=2.7.0' 'aiohttp>=3.14.3' 'pyarrow>=23.0.1' 'cryptography>=50.0.0' 'pyasn1>=0.6.4' 'pillow>=12.3.0' 'msgpack>=1.2.1' 'setuptools>=83.0.0'

# base-env transitive deps not reached by the ptca pip installs; explicit upgrades required:
# python-dotenv>=1.2.2 (GHSA-mf9w-mj56-hr94), urllib3>=2.7.0 (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j),
# idna>=3.15 (GHSA-65pc-fj4g-8rjx), click>=8.3.3 (GHSA-47fr-3ffg-hgmw), aiohttp>=3.14.3
# (GHSA-hg6j-4rv6-33pg, GHSA-jg22-mg44-37j8, GHSA-mfx4-hv73-q22v, GHSA-cq5v-8q36-5273,
# GHSA-mq44-7p77-q5h7), PyJWT>=2.13.0 (GHSA-jq35-7prp-9v3f, GHSA-993g-76c3-p5m4).
# py-rattler: pinned transitive dep of ACPT base image conda tooling; brought in by base image, fixes GHSA-q53q-5r4j-5729.
# cryptography: pinned transitive dep of ACPT base image; brought in by base image, fixes
#   GHSA-537c-gmf6-5ccf and GHSA-g6cj-pr64-35w5.
# pydantic-settings: pinned transitive dep of ACPT base image; brought in by base image, fixes GHSA-4xgf-cpjx-pc3j.
# pyasn1: transitive dep of mlflow -> databricks-sdk -> google-auth; fixes GHSA-hm4w-wwcw-mr6r and GHSA-8ppf-4f7h-5ppj.
# setuptools: bootstrapped by the ACPT base conda image; fixes GHSA-h35f-9h28-mq5c.
# msgpack: the base conda prefix carries its own copy that the ptca pin above never reaches;
#   fixes GHSA-6v7p-g79w-8964.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'idna>=3.15' 'click>=8.3.3' 'aiohttp>=3.14.3' 'PyJWT>=2.13.0' 'py-rattler>=0.24.0' 'cryptography>=50.0.0' 'pydantic-settings>=2.14.2' 'pyasn1>=0.6.4' 'setuptools>=83.0.0' 'msgpack>=1.2.1'

# pip vendors its own copies of setuptools and msgpack and declares them in
# pip/_vendor/bom.cdx.json, which the scanner reports as setuptools 70.3.0
# (GHSA-5rjg-fvgr-3xxf) and msgpack 1.1.2 (GHSA-6v7p-g79w-8964) even though the
# real packages above are patched. Drop the stale metadata.
RUN set -eux; \
    find / -xdev -path '*/site-packages/pip/_vendor/bom.cdx.json' -type f -delete; \
    find /opt /root -type d \( -name 'setuptools-70.3.0.dist-info' -o -name 'msgpack-1.1.2.dist-info' \) -prune -exec rm -rf {} +; \
    find /opt /root -path '*/_vendor/vendor.txt' -type f -exec \
        sed -i -E 's/setuptools==70\.3\.0/setuptools==83.0.0/g; s/msgpack==1\.1\.2/msgpack==1.2.1/g' {} +

# torch: pinned dep from ACPT ptca env; fixes GHSA-rrmf-rvhw-rf47, GHSA-qfhq-4f3w-5fph, and GHSA-vgrw-7cvw-pwgx.
# torchvision and torchaudio are upgraded with torch to keep the CUDA PyTorch stack compatible.
RUN pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0'
