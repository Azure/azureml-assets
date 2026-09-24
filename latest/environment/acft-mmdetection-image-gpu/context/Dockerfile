FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202609.2

USER root

# OS packages inherited from the ACPT base image but still below current
# Ubuntu security versions; fixes USN-8678-1, USN-8670-2, USN-8651-1, USN-8675-1,
# USN-8687-1, USN-8691-1, USN-8699-1, USN-8692-1, USN-8697-1, USN-8711-1,
# USN-8710-1, USN-8721-1, USN-8736-1, USN-8737-1, USN-8740-1, and USN-8744-1.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade \
    libssl-dev=3.0.2-0ubuntu1.29 \
    libssl3=3.0.2-0ubuntu1.29 \
    openssl=3.0.2-0ubuntu1.29 \
    libcurl4=7.81.0-1ubuntu1.27 \
    curl=7.81.0-1ubuntu1.27 \
    libcurl3-gnutls=7.81.0-1ubuntu1.27 \
    libperl5.34=5.34.0-3ubuntu1.9 \
    perl=5.34.0-3ubuntu1.9 \
    perl-modules-5.34=5.34.0-3ubuntu1.9 \
    perl-base=5.34.0-3ubuntu1.9 \
    libp11-kit0=0.24.0-6ubuntu0.1 \
    libattr1=1:2.5.1-1ubuntu0.1 \
    libssh-4=0.9.6-2ubuntu0.22.04.8 \
    diffutils=1:3.8-0ubuntu2.1 \
    coreutils=8.32-4.1ubuntu1.4 \
    libgcrypt20=1.9.4-3ubuntu3.3 \
    libevent-core-2.1-7=2.1.12-stable-1ubuntu0.1 \
    openssh-client=1:8.9p1-3ubuntu0.17 \
    libc6-dev=2.35-0ubuntu3.15 \
    libc-bin=2.35-0ubuntu3.15 \
    libc6=2.35-0ubuntu3.15 \
    libc-dev-bin=2.35-0ubuntu3.15 \
    locales=2.35-0ubuntu3.15 \
    dotnet-runtime-8.0=8.0.31-0ubuntu1~22.04.1 \
    dotnet-hostfxr-8.0=8.0.31-0ubuntu1~22.04.1 \
    dotnet-host-8.0=8.0.31-0ubuntu1~22.04.1 \
    python3.10=3.10.12-1~22.04.18 \
    python3.10-minimal=3.10.12-1~22.04.18 \
    libpython3.10-minimal=3.10.12-1~22.04.18 \
    libpython3.10-stdlib=3.10.12-1~22.04.18 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# openmim still imports pkg_resources, which is removed from setuptools>=82.
# Keep a temporary compatibility pin for mim installation, then restore setuptools.
# Note: mmdet should be installed via mim to access the model zoo config folder.
# Temporary workaround for https://github.com/open-mmlab/mmdetection/issues/11668 (when mmdet updated, remove lines below)
# setuptools: pinned inherited dep from the ACPT base image; fixes GHSA-h35f-9h28-mq5c.
RUN pip install -r requirements.txt --no-cache-dir \
 && /opt/conda/bin/python -m pip install --no-cache-dir 'setuptools<82' \
 && conda run -n ptca python -m pip install --no-cache-dir 'setuptools<82' \
 && mim install mmdet==3.3.0 \
 && mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html --no-cache-dir \
 && /opt/conda/bin/python -m pip install --no-cache-dir --upgrade 'setuptools==83.0.0' \
 && conda run -n ptca python -m pip install --no-cache-dir --upgrade 'setuptools==83.0.0' \
 && sed -i 's/2.2.0/2.3.0/' /opt/conda/envs/ptca/lib/python3.10/site-packages/mmdet/__init__.py

# onnx-weekly: pinned transitive dep of azureml-acft-accelerator; it caps
# onnx<=1.17.0, fixes GHSA-3r9x-f23j-gc73 and GHSA-hqmj-h5c6-369m.
RUN pip uninstall -y onnx && pip install --no-cache-dir 'onnx-weekly>=1.22.0.dev20260504'

# pyarrow: pinned transitive dep of datasets; image install downgrades the base, fixes CVE-2026-25087.
# aiohttp: pinned inherited dep from the ACPT base image; fixes GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
# cryptography: pinned inherited base dep and transitive dep of azureml-core in ptca; fixes GHSA-g6cj-pr64-35w5.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.3' \
    'cryptography>=50.0.0' \
 && conda run -n ptca python -m pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.3' \
    'cryptography>=50.0.0'
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
    'pyarrow>=23.0.1'

# torch: pinned transitive dep of the ACFT image stack; this image overrides
# the ACPT base version, fixes GHSA-rrmf-rvhw-rf47.
RUN conda run -n ptca python -m pip uninstall -y torchaudio \
 && conda run -n ptca python -m pip install --no-cache-dir --upgrade \
    --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.13.0' \
    'torchvision==0.28.0'

# The conda pip upgrade can reinstall the rattler solver. The base image does
# not need it for runtime, and its compiled extension currently carries Rust CVEs.
RUN conda config --system --set solver classic \
 && /opt/conda/bin/python3.13 -m pip uninstall -y py-rattler conda-rattler-solver 2>/dev/null || true \
 && rm -rf /opt/conda/lib/python3.13/site-packages/rattler* /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* /opt/conda/lib/python3.13/site-packages/py_rattler* /opt/conda/conda-meta/py-rattler-*.json /opt/conda/conda-meta/conda-rattler-solver-*.json

# pip vendor metadata keeps stale bundled package versions in the SBOM after pip upgrades.
RUN find /opt/conda -path '*/site-packages/pip/_vendor/bom.cdx.json' -delete \
 && find /opt/conda -path '*/site-packages/pip/_vendor/vendor.txt' -delete

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
