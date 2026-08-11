# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root
RUN apt-get -y update \
 && apt-get install -y --only-upgrade \
    libc-dev-bin libc6-dev locales libc6 libc-bin \
    libpam-runtime libpam0g libpam-modules libpam-modules-bin \
    libkrb5support0 libgssapi-krb5-2 libk5crypto3 libkrb5-3 \
    tar wget libsqlite3-0 openssh-client \
    libasound2 libasound2-data \
    dotnet-hostfxr-8.0 dotnet-runtime-8.0 dotnet-host-8.0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# setuptools: pinned from the ACPT base conda env; fixes GHSA-h35f-9h28-mq5c.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade 'setuptools>=83.0.0'

# aiohttp and cryptography: pinned transitive deps inherited from the ACPT base conda env; fixes GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, GHSA-mfx4-hv73-q22v, and GHSA-g6cj-pr64-35w5.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade \
    'aiohttp>=3.14.3' \
    'cryptography>=50.0.0,<51'

# transformers: fix GHSA-69w3-r845-3855 (CVE-2026-1839, arbitrary code execution in Trainer).
# --no-deps keeps the pinned HF stack intact (transformers is a direct requirements.txt pin).
RUN pip install --no-cache-dir --no-deps 'transformers==5.5.4'

# Compatibility pins for the MMTracking stack.
RUN pip install scipy==1.10.1
RUN pip install gunicorn==23.0.0

# onnx: pinned transitive dep of azureml-acft-accelerator; keeps the base onnx from being downgraded and fixes GHSA-hwpq-hmq9-wj77.
RUN pip install --no-cache-dir --upgrade 'onnx==1.22.0'

# cryptography: pinned transitive dep of azureml-core/pyOpenSSL; fixes GHSA-g6cj-pr64-35w5.
# GitPython: pinned transitive dep of azureml-core; fixes GHSA-94p4-4cq8-9g67, GHSA-fjr4-x663-mwxc, GHSA-6p8h-3wgx-97gf, GHSA-r9mr-m37c-5fr3, GHSA-p538-c434-8v24, GHSA-539m-9xh6-q6rr, and GHSA-3f7w-8rr8-f37f.
# aiohttp: pinned transitive dep inherited from the ACPT base image; fixes GHSA-mq44-7p77-q5h7, GHSA-cq5v-8q36-5273, and GHSA-mfx4-hv73-q22v.
RUN pip install --no-cache-dir --upgrade \
    'cryptography>=50.0.0,<51' \
    'GitPython>=3.1.57,<4' \
    'aiohttp>=3.14.3'

# It is recommended to upgrade to numpy version 1.23.5 to address the numpy.core.multiarray issue
RUN pip install numpy==1.23.5

# pin required version of yapf to avoid breaking changes in newer versions
# https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install yapf==0.40.1

# diffusers 0.24.0 -> 0.38.0 to fix GHSA-98h9-4798-4q5v (CVE-2026-44513).
# --no-deps keeps the pinned HF stack intact (diffusers is a direct requirements.txt pin).
RUN pip install --no-cache-dir --no-deps 'diffusers==0.38.0'

# pyarrow>=23.0.1 in ptca env (py3.10): fix GHSA-rgxp-2hwp-jwgg (API-compatible with 20.x).
# pinned transitive dep of datasets; fixes GHSA-rgxp-2hwp-jwgg.
RUN pip install --no-cache-dir 'pyarrow>=23.0.1'

# torch: pinned dep from ACPT ptca env; fixes GHSA-rrmf-rvhw-rf47, GHSA-qfhq-4f3w-5fph, and GHSA-vgrw-7cvw-pwgx.
# torchvision and torchaudio are upgraded with torch to keep the CUDA PyTorch stack compatible.
RUN pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0'
