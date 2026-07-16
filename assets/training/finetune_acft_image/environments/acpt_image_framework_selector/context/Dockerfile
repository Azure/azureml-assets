# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

# Pull latest Ubuntu 22.04 security patches on each rebuild.
# Explicit package list ensures frequently flagged USN fixes are applied even
# when apt-get upgrade would otherwise hold them back.
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --only-upgrade \
        curl \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        libcurl3-gnutls \
        libcurl4 \
        libgcrypt20 \
        liblzma5 \
        libnghttp2-14 \
        libsqlite3-0 \
        xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages (ptca env). setuptools>=82.0.1 patches
# GHSA-58pv-8j8x-9vj2 in the ptca env which ships setuptools 81.0.0.
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Flag needed to enable control flow which is in PrP.
ENV AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=True

# Vulnerability overrides in base conda env. The base SBOM is patched for
# several of these packages, but requirements.txt can re-resolve
# loose AzureML/MLflow transitive dependencies back to vulnerable versions.
# - setuptools: base ships 82.0.0; need >=82.0.1 for GHSA-58pv-8j8x-9vj2.
# - python-dotenv>=1.2.2: transitive dep of pydantic-settings and
#   anaconda-auth; fixes GHSA-mf9w-mj56-hr94.
# - urllib3>=2.7.0: transitive dep of requests; fixes GHSA-qccp-gfcp-xxvc and
#   GHSA-mf9v-mfxr-j63j.
# - aiohttp>=3.14.0: fixes GHSA-jg22-mg44-37j8 and GHSA-hg6j-4rv6-33pg.
# - click>=8.3.3: transitive dep of typer/mlflow-skinny; fixes
#   GHSA-47fr-3ffg-hgmw.
# - idna>=3.15: transitive dep of httpx and requests; fixes
#   GHSA-65pc-fj4g-8rjx.
# - PyJWT>=2.13.0: transitive dep from the base image; fixes
#   GHSA-993g-76c3-p5m4 and GHSA-jq35-7prp-9v3f.
# - py-rattler: pinned transitive dep of base conda/rattler tooling brought in
#   by the base image; fixes GHSA-q53q-5r4j-5729.
# - cryptography: pinned transitive dep brought in by the ACPT base image;
#   fixes GHSA-537c-gmf6-5ccf.
# - pydantic-settings: pinned transitive dep of azureml-defaults ->
#   azureml-inference-server-http; brought in by requirements.txt, fixes
#   GHSA-4xgf-cpjx-pc3j.
# - msgpack: pinned transitive dep of azureml-mlflow -> mlflow-skinny; brought
#   in by requirements.txt, fixes GHSA-6v7p-g79w-8964.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
        'setuptools>=82.0.1' 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' \
        'aiohttp>=3.14.0' 'click>=8.3.3' 'idna>=3.15' 'PyJWT>=2.13.0' \
        'py-rattler>=0.24.0' 'cryptography>=48.0.1' 'pydantic-settings>=2.14.2' \
        'msgpack>=1.2.1'

# Vulnerability overrides in ptca conda env (python3.10):
# - urllib3: requirements can re-resolve loose dependencies back to vulnerable
#   urllib3 versions. Parents (requests, torchdata, fsspec) use loose floors, so
#   they cannot force urllib3>=2.7.0; direct override is required.
# - aiohttp: ptca env ships 3.13.5; need >=3.14.0 for GHSA-jg22-mg44-37j8 and
#   GHSA-hg6j-4rv6-33pg. No parent pins aiohttp to a safe floor; direct
#   override is required.
# - starlette: ptca env ships 0.52.1; need >=1.0.1 for GHSA-86qp-5c8j-p5mr.
#   No parent in the ptca env forces starlette>=1.0.1; direct override is
#   required.
# - click: ptca env ships 8.2.1; need >=8.3.3 for GHSA-47fr-3ffg-hgmw. No
#   parent in the ptca env forces click>=8.3.3; direct override is required.
# - pyarrow: ptca env ships 20.0.0; need >=23.0.1 for GHSA-rgxp-2hwp-jwgg.
#   No parent in the ptca env forces pyarrow>=23.0.1; direct override is
#   required.
# - cryptography: pinned transitive dep brought in by the ACPT base image;
#   fixes GHSA-537c-gmf6-5ccf.
# - torch: inherited from the ACPT torch280 base image; fixes
#   GHSA-vgrw-7cvw-pwgx and GHSA-qfhq-4f3w-5fph. torchvision and torchaudio are
#   upgraded with torch to keep the CUDA PyTorch stack compatible.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
        'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'starlette>=1.0.1' 'click>=8.3.3' \
        'pyarrow>=23.0.1' 'cryptography>=48.0.1' && \
    conda run -n ptca python -m pip install --no-cache-dir --upgrade \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
