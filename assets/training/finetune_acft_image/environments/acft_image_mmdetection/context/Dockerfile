FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

USER root

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# openmim still imports pkg_resources, which is removed from setuptools>=82.
# Keep a temporary compatibility pin for mim installation, then restore setuptools.
RUN pip install --no-cache-dir 'setuptools<82'

# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0
# Temporary workaround for https://github.com/open-mmlab/mmdetection/issues/11668 (when mmdet updated, remove lines below)
RUN mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html --no-cache-dir
RUN pip install --no-cache-dir --upgrade setuptools==82.0.0
RUN sed -i 's/2.2.0/2.3.0/' /opt/conda/envs/ptca/lib/python3.10/site-packages/mmdet/__init__.py

# onnx-weekly: pinned transitive dep of azureml-acft-accelerator; it caps
# onnx<=1.17.0, fixes GHSA-3r9x-f23j-gc73 and GHSA-hqmj-h5c6-369m.
RUN pip uninstall -y onnx && pip install --no-cache-dir 'onnx-weekly>=1.22.0.dev20260504'

# pyarrow: pinned transitive dep of datasets; image install downgrades the base, fixes CVE-2026-25087.
# cryptography: pinned transitive dep of azureml-core; image install downgrades the base, fixes GHSA-537c-gmf6-5ccf.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
    'pyarrow>=23.0.1' \
    'cryptography>=48.0.1'

# torch: pinned inherited dep from the ACPT ptca base image; fixes
# CVE-2025-2999 and CVE-2025-3001. Upgrade the CUDA PyTorch stack together.
RUN conda run -n ptca python -m pip install --no-cache-dir --upgrade \
    --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.10.0' \
    'torchvision==0.25.0' \
    'torchaudio==2.10.0'

# The conda pip upgrade can reinstall the rattler solver. The base image does
# not need it for runtime, and its compiled extension currently carries Rust CVEs.
RUN conda config --system --set solver classic \
 && /opt/conda/bin/python3.13 -m pip uninstall -y py-rattler conda-rattler-solver 2>/dev/null || true \
 && rm -rf /opt/conda/lib/python3.13/site-packages/rattler* /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* /opt/conda/lib/python3.13/site-packages/py_rattler* /opt/conda/conda-meta/py-rattler-*.json /opt/conda/conda-meta/conda-rattler-solver-*.json

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
