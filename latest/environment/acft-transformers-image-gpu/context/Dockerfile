# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.2

USER root
# Pull latest Ubuntu 22.04 security patches.
# Explicit --only-upgrade list ensures USN-8298-1 (.NET), USN-8319-1 (libgcrypt20),
# and USN-8362-1 (xz-utils/liblzma5) patches are applied; || true skips packages
# not present in this image without failing the build.
RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install --only-upgrade \
        dotnet-hostfxr-8.0 \
        dotnet-host-8.0 \
        dotnet-runtime-8.0 \
        libgcrypt20 \
        liblzma5 \
        xz-utils \
 || true \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Override vulnerable transitive deps in ptca env that are not fixed in the base image
# onnx: azureml-acft-accelerator==0.0.89 requires onnx<=1.17.0 which downgrades base 1.21.0;
#   override needed to keep safe version (GHSA-p433-9wv8-28xj etc.)
# pyasn1: transitive dep (mlflow → databricks-sdk → google-auth → pyasn1-modules → pyasn1);
#   parent packages use loose floors so pip resolves to 0.6.2 which has CVE-2026-30922; override to >=0.6.3
# fastmcp: transitive dep (mlflow-skinny[mcp] requires fastmcp<4,>=2.0.0); loose floor,
#   GHSA-rww4-4w9c-7733, GHSA-m8x7-r2rg-vh5g, GHSA-vv7q-7jx5-f767; override to >=3.2.0
# Mako: transitive dep (mlflow → alembic → Mako); alembic 1.18.4 requires Mako (no version pin),
#   pip won't upgrade pre-installed Mako; override needed (GHSA-v92g-xgxw-vvmm)
# GitPython: transitive dep (mlflow → mlflow-skinny requires gitpython<4,>=3.1.9); loose floor,
#   pip resolves to 3.1.46 which has GHSA-rpm5-65cw-6hj4, GHSA-x2qx-6953-8485; override to >=3.1.47
# python-dotenv: transitive dep (mlflow → mlflow-skinny requires python-dotenv<2,>=0.19.0); loose floor,
#   pip resolves to 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
# pillow: comes from the base ACPT image (not a hard runtime dep of any requirements.txt package;
#   transformers/datasets/diffusers/optimum/huggingface-hub only require Pillow under extras_require
#   like [vision]/[dev]/[testing]); base image ships 12.1.1 which has GHSA-whj4-6x5x-4v2j;
#   no parent package to upgrade — explicit override required (>=12.2.0)
# pytest: comes from the base ACPT image (not a hard runtime dep of any requirements.txt package;
#   azureml-acft-accelerator only pins pytest~=5.3.0 under extras_require [test]); base image ships
#   7.4.3 which has GHSA-6w46-j5rx-g56g; no parent package to upgrade — explicit override required (>=9.0.3)
# urllib3: transitive dep of requests/botocore/etc.; parent packages use loose
#   floors (requests>=2.33 allows urllib3<3,>=1.21.1) so pip won't pull a newer
#   urllib3 on its own. Base image still ships 2.6.3 in the base conda env,
#   vulnerable to GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j; override to >=2.7.0.
# aiohttp: transitive dep (mlflow and other packages); ships 3.13.5 which has
#   GHSA-jg22-mg44-37j8 and GHSA-hg6j-4rv6-33pg; handled separately via conda run
#   (see below) to ensure the correct ptca env is targeted.
RUN pip install --no-cache-dir --upgrade 'onnx>=1.21.0' pyasn1==0.6.3 'fastmcp>=3.2.0' 'Mako>=1.3.12' 'GitPython>=3.1.47' 'python-dotenv>=1.2.2' 'pillow>=12.2.0' 'pytest>=9.0.3' 'urllib3>=2.7.0'
# pip: package installer itself — there is no parent package that brings it in.
#   Both conda envs (ptca python3.10, base python3.13) ship pip 26.0.1 from the base
#   image, vulnerable to GHSA-jp4c-xjxw-mgf9. Use conda-forge to upgrade pip in both
#   envs without allowing the solver to roll back other packages (--freeze-installed).
RUN conda install -n ptca -y -c conda-forge --freeze-installed 'pip=26.1.1' \
 && conda install -n base -y -c conda-forge --freeze-installed 'pip=26.1.1'
# python-dotenv in base conda env: transitive dep of uvicorn[standard] (>=0.13); loose floor,
#   base image has 1.2.1 which has GHSA-mf9w-mj56-hr94; override to >=1.2.2
# urllib3 in base conda env: same root cause as above — base ships 2.6.3 via
#   requests/botocore transitive chain; override to >=2.7.0 for
#   GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j.
# aiohttp in base conda env: base image ships 3.13.5 which has GHSA-jg22-mg44-37j8
#   and GHSA-hg6j-4rv6-33pg; override to >=3.14.0.
# click in base conda env: base image ships 8.2.1 which has GHSA-47fr-3ffg-hgmw;
#   no parent package to upgrade — explicit override required (>=8.3.3).
# idna in base conda env: base image ships 3.11 which has GHSA-65pc-fj4g-8rjx;
#   no parent package to upgrade — explicit override required (>=3.15).
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2' 'urllib3>=2.7.0' 'aiohttp>=3.14.0' 'click>=8.3.3' 'idna>=3.15'
# aiohttp in ptca env: same CVEs as base env — GHSA-jg22-mg44-37j8 and GHSA-hg6j-4rv6-33pg.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade 'aiohttp>=3.14.0'
# The conda pip upgrade can reinstall the rattler solver. The base image does
# not need it for runtime, and its compiled extension currently carries Rust CVEs.
RUN conda config --system --set solver classic \
 && /opt/conda/bin/python3.13 -m pip uninstall -y py-rattler conda-rattler-solver 2>/dev/null || true \
 && rm -rf /opt/conda/lib/python3.13/site-packages/rattler* \
           /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* \
           /opt/conda/lib/python3.13/site-packages/py_rattler* \
           /opt/conda/conda-meta/py-rattler-*.json \
           /opt/conda/conda-meta/conda-rattler-solver-*.json
RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
