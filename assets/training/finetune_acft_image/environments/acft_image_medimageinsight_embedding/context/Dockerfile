FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

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
        openssh-client \
        python3.10 \
        python3.10-minimal \
        tar \
        wget && \
    apt-get install -y --no-install-recommends unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --progress-bar off -r requirements.txt --no-cache-dir

# ptca env torch is inherited from the ACPT torch280 base image; pin the patched CUDA build for
# GHSA-qfhq-4f3w-5fph, GHSA-vgrw-7cvw-pwgx, and GHSA-rrmf-rvhw-rf47. Keep torchvision aligned
# with torch; remove torchaudio because no compatible 2.13 CUDA wheel is available.
RUN /opt/conda/envs/ptca/bin/python -m pip install --no-cache-dir --upgrade \
        --progress-bar off \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.13.0+cu126' \
        'torchvision==0.28.0+cu126' && \
    /opt/conda/envs/ptca/bin/python -m pip uninstall -y torchaudio && \
    rm -rf /opt/conda/envs/ptca/lib/python3.10/site-packages/torch-2.8*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torch-2.10*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchvision-0.23*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchvision-0.25*.dist-info \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchaudio \
           /opt/conda/envs/ptca/lib/python3.10/site-packages/torchaudio-*.dist-info

# mlflow 3.5.0 has CVEs (CVE-2025-14287, CVE-2026-2033, CVE-2026-2635, GHSA-fh64-r2vc-xvhr);
# upgrade after requirements install. azureml-mlflow 1.62.0.post2 pins mlflow-skinny<=3.9.0,
# so mlflow must be upgraded separately (post-pip-install) to avoid resolver conflict.
# mlflow 3.15.0 additionally fixes GHSA-7gwp-5pfp-969j and GHSA-2cm6-r77w-6g96.
RUN pip install --progress-bar off --no-cache-dir mlflow==3.15.0
# fastmcp + mcp were installed as regular deps of mlflow 3.5.0 (fastmcp -> mcp) but orphaned after
# mlflow 3.11.1 (fastmcp moved to optional "mcp" extra); uninstall both to remove vulnerable packages
RUN pip uninstall -y fastmcp mcp

# Override vulnerable transitive deps in the ptca env (Python 3.10) that pip won't auto-upgrade:
# Mako: transitive dep (mlflow -> alembic 1.18.4 -> Mako); alembic 1.18.4 declares "Mako" with no
#   version pin at all, so pip resolves to 1.3.10 which has GHSA-v92g-xgxw-vvmm. No parent
#   release floors Mako>=1.3.11, so explicit override is required.
# GitPython: transitive dep (mlflow -> mlflow-skinny requires gitpython<4,>=3.1.9); the
#   loose floor can resolve vulnerable builds; override to >=3.1.57 for GHSA-2f96-g7mh-g2hx,
#   GHSA-956x-8gvw-wg5v, GHSA-v396-v7q4-x2qj, GHSA-94p4-4cq8-9g67,
#   GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, GHSA-r9mr-m37c-5fr3,
#   GHSA-p538-c434-8v24, GHSA-539m-9xh6-q6rr, and GHSA-3f7w-8rr8-f37f.
# aiohttp: transitive dep; lower versions have GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg,
#   GHSA-mfx4-hv73-q22v, GHSA-cq5v-8q36-5273, and GHSA-mq44-7p77-q5h7.
#   No parent release floors aiohttp>=3.14.3, so explicit override is required.
# pyarrow: resolves to 20.0.0 which has GHSA-rgxp-2hwp-jwgg; no parent release floors
#   pyarrow>=23.0.1, so explicit override is required.
# cryptography: transitive dep of ACPT base image and AzureML requirements; fixes GHSA-537c-gmf6-5ccf
#   and GHSA-g6cj-pr64-35w5.
# pyasn1: transitive dep of mlflow -> databricks-sdk -> google-auth; fixes GHSA-hm4w-wwcw-mr6r and GHSA-8ppf-4f7h-5ppj.
# pillow: transitive dep of image/vision packages; fixes GHSA-8v84-f9pq-wr9x, GHSA-9hw9-ch79-4vh6,
#   GHSA-vjc4-5qp5-m44j, GHSA-pg7v-jwj7-p798, GHSA-45hq-cxwh-f6vc, GHSA-phj9-mv4w-65pm,
#   GHSA-62p4-gmf7-7g93, and GHSA-5x94-69rx-g8h2.
# sqlparse: transitive dep (mlflow -> sqlalchemy tooling); no parent release floors
#   sqlparse>=0.6.0, so explicit override is required. Fixes GHSA-f2ff-p2ww-7p4p,
#   GHSA-prg7-hcfm-mfcr, GHSA-3496-9g83-7v6x, and GHSA-pwgv-4x5q-6m9f.
RUN pip install --progress-bar off --no-cache-dir --upgrade 'Mako>=1.3.11' 'GitPython>=3.1.57' 'aiohttp>=3.14.3' 'pyarrow>=23.0.1' 'cryptography>=50.0.0' 'pyasn1>=0.6.4' 'pillow>=12.3.0' 'sqlparse>=0.6.0'

# Upgrade vulnerable packages in the base conda Python
# urllib3: transitive dep via requests; loose floor (urllib3<3,>=1.21.1) resolves to vulnerable 2.6.3
#   (GHSA-qccp-gfcp-xxvc, GHSA-mf9v-mfxr-j63j). No parent release floors urllib3>=2.7.0, override required.
# python-dotenv: transitive dep via pydantic-settings (>=0.21.0 floor); parent uses loose floor so base resolves to
#   vulnerable 1.2.1 (GHSA-mf9w-mj56-hr94); override to >=1.2.2
# idna: transitive via requests (idna<4,>=2.5), anyio (idna>=2.8), httpx (no pin) - all loose floors
#   resolve to vulnerable 3.11 (GHSA-65pc-fj4g-8rjx / CVE-2026-45409). No parent release floors
#   idna>=3.15, so explicit override is required.
# click: transitive via mlflow-skinny (click<9,>=7.0), uvicorn (click>=7.0), typer (click>=8.2.1),
#   flask (click>=8.1.3) - all loose floors resolve to vulnerable 8.2.1 (GHSA-47fr-3ffg-hgmw /
#   CVE-2026-7246). No parent release floors click>=8.3.3, so explicit override is required.
# aiohttp: lower versions have GHSA-jg22-mg44-37j8, GHSA-hg6j-4rv6-33pg,
#   GHSA-mfx4-hv73-q22v, GHSA-cq5v-8q36-5273, and GHSA-mq44-7p77-q5h7. No parent
#   release floors aiohttp>=3.14.3, so explicit override is required.
# PyJWT: resolves to 2.12.1 which has GHSA-jq35-7prp-9v3f and GHSA-993g-76c3-p5m4. No parent
#   release floors PyJWT>=2.13.0, so explicit override is required.
# setuptools: bootstrapped by the ACPT base conda image; fixes GHSA-h35f-9h28-mq5c.
# py-rattler: transitive dep of ACPT base image conda tooling; fixes GHSA-q53q-5r4j-5729.
# cryptography: transitive dep of ACPT base image and AzureML requirements; fixes GHSA-537c-gmf6-5ccf
#   and GHSA-g6cj-pr64-35w5.
# pydantic-settings: transitive dep of AzureML requirements that can downgrade the patched base; fixes GHSA-4xgf-cpjx-pc3j.
RUN /opt/conda/bin/python -m pip install --progress-bar off --no-cache-dir --upgrade 'urllib3>=2.7.0' 'python-dotenv>=1.2.2' 'idna>=3.15' 'click>=8.3.3' 'aiohttp>=3.14.3' 'PyJWT>=2.13.0' 'setuptools>=83.0.0' 'py-rattler>=0.24.0' 'cryptography>=50.0.0' 'pydantic-settings>=2.14.2' && \
    rm -rf /opt/conda/lib/python3.14/site-packages/setuptools-82.0.0.dist-info
