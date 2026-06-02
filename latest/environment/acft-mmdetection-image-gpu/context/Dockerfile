FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.1

USER root
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# openmim still imports pkg_resources, which is removed from setuptools>=82.
# Keep a temporary compatibility pin for mim installation, then restore setuptools.
RUN pip install --no-cache-dir 'setuptools<82'

# # # Install mmdet
# # Note that MMDet installs pycocotools
# Note: mmdet should be installed via mim to access the model zoo config folder.
RUN mim install mmdet==3.3.0
# Temporary workaround for https://github.com/open-mmlab/mmdetection/issues/11668 (when mmdet updated, remove lines below)
RUN mim install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html --no-cache-dir
RUN pip install --no-cache-dir --upgrade setuptools==82.0.0
RUN sed -i 's/2.2.0/2.3.0/' /opt/conda/envs/ptca/lib/python3.10/site-packages/mmdet/__init__.py

# azureml-acft-accelerator still pins onnx<=1.17.0, which keeps vulnerable onnx
# in the image. Use onnx-weekly until that parent package allows a patched stable onnx.
RUN pip uninstall -y onnx && pip install --no-cache-dir 'onnx-weekly>=1.22.0.dev20260504'

# pip has no parent package to upgrade. The current base still ships pip 26.0.1
# in both Python envs, so install the patched conda-forge build without changing
# the rest of the environment.
RUN conda install -n ptca -y -c conda-forge --freeze-installed 'pip=26.1.1' \
 && conda install -n base -y -c conda-forge --freeze-installed 'pip=26.1.1'

# Base env packages are outside requirements.txt. Metadata checks show their
# parents allow patched versions but do not require them, so keep direct floors.
# urllib3 is pulled by conda/requests-style tooling and must be >=2.7.0 for
# GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j.
RUN conda run -n base python -m pip install --no-cache-dir --upgrade \
    'urllib3>=2.7.0,<3' \
    'idna>=3.15' \
    'click>=8.3.3' \
    'python-dotenv>=1.2.2'

# The conda pip upgrade can reinstall the rattler solver. The base image does
# not need it for runtime, and its compiled extension currently carries Rust CVEs.
RUN conda config --system --set solver classic \
 && /opt/conda/bin/python3.13 -m pip uninstall -y py-rattler conda-rattler-solver 2>/dev/null || true \
 && rm -rf /opt/conda/lib/python3.13/site-packages/rattler* /opt/conda/lib/python3.13/site-packages/conda_rattler_solver* /opt/conda/lib/python3.13/site-packages/py_rattler* /opt/conda/conda-meta/py-rattler-*.json /opt/conda/conda-meta/conda-rattler-solver-*.json

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
