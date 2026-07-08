# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install --only-upgrade \
      liblzma5 \
      xz-utils \
      libgcrypt20 \
      dotnet-host-8.0 \
      dotnet-hostfxr-8.0 \
      dotnet-runtime-8.0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# transformers: fix GHSA-69w3-r845-3855 (CVE-2026-1839, arbitrary code execution in Trainer).
# --no-deps keeps the pinned HF stack intact (transformers is a direct requirements.txt pin).
RUN pip install --no-cache-dir --no-deps 'transformers==5.5.4'

# vulnerability fixes - this will be removed once we update to MMTracking's latest version
RUN pip install scipy==1.10.1
RUN pip install gunicorn==23.0.0
RUN pip install onnx==1.21.0

# It is recommended to upgrade to numpy version 1.23.5 to address the numpy.core.multiarray issue
RUN pip install numpy==1.23.5

# pin required version of yapf to avoid breaking changes in newer versions
# https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install yapf==0.40.1

# cryptography: pinned transitive dep of base ACPT image and AzureML SDK dependencies; fixes GHSA-537c-gmf6-5ccf.
RUN pip install --no-cache-dir --upgrade 'cryptography>=48.0.1,<49' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'cryptography>=48.0.1,<49'

# diffusers 0.24.0 -> 0.38.0 to fix GHSA-98h9-4798-4q5v (CVE-2026-44513).
# --no-deps keeps the pinned HF stack intact (diffusers is a direct requirements.txt pin).
RUN pip install --no-cache-dir --no-deps 'diffusers==0.38.0'

# pyarrow>=23.0.1 in ptca env (py3.10): fix GHSA-rgxp-2hwp-jwgg (API-compatible with 20.x).
# Transitive dep with no parent pinning a safe floor.
RUN pip install --no-cache-dir 'pyarrow>=23.0.1'

# torch==2.10.0: pinned dep from ACPT ptca env; fixes GHSA-qfhq-4f3w-5fph and GHSA-vgrw-7cvw-pwgx.
# torchvision and torchaudio are upgraded with torch to keep the CUDA PyTorch stack compatible.
RUN pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0'
