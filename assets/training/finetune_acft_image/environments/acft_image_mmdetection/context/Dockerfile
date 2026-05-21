# PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}

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

# onnx: azureml-acft-accelerator pins onnx<=1.17.0 but that range has known CVEs;
#   onnx-weekly 1.22.0.dev20260504 includes main-branch security fixes (GHSA-3r9x-f23j-gc73,
#   GHSA-538c-55jv-c5g9, GHSA-hqmj-h5c6-369m, etc.) not yet in a stable PyPI release (checked 2026-05-12)
RUN pip uninstall -y onnx && pip install --no-cache-dir 'onnx-weekly>=1.22.0.dev20260504'
# pip 26.0.1 (GHSA-jp4c-xjxw-mgf9): pip is the Python package installer itself with no upstream parent package,
#   so direct upgrade is the only remediation. pip 26.1+ is on PyPI and conda-forge; the conda `defaults` channel
#   currently tops out at 26.0.1 (checked 2026-05-12), hence `-c conda-forge` is required. Using `conda install`
#   (not `pip install`) so both the dist-info METADATA and the conda-meta JSON are refreshed in one step,
#   ensuring SBOM scanners pick up the new version. `--freeze-installed` keeps the rest of the env intact.
RUN conda install -n ptca -y -c conda-forge --freeze-installed 'pip=26.1.1'
# python-dotenv 1.2.1 (GHSA-mf9w-mj56-hr94): brought in by azureml-inference-server-http -> pydantic-settings -> python-dotenv>=0.21.0;
#   parent packages do not upper-bound python-dotenv so upgrading them won't force >=1.2.2; direct override required (checked 2026-05-12)
RUN conda run -n base python -m pip install --no-cache-dir --upgrade 'python-dotenv>=1.2.2'
# pip 26.0.1 (GHSA-jp4c-xjxw-mgf9) also present in base env (Python 3.13); same direct-upgrade rationale as ptca env above.
RUN conda install -n base -y -c conda-forge --freeze-installed 'pip=26.1.1'

# Override diffusers 0.24.0 -> 0.38.0 to fix GHSA-98h9-4798-4q5v (CVE-2026-44513).
# Root cause (verified 2026-05 via PyPI metadata against the built image):
#   - diffusers is directly pinned in requirements.txt line 11 at ==0.24.0 alongside the rest
#     of the pinned HF stack (transformers==5.5.4, accelerate==0.27.2, optimum==1.23.3,
#     huggingface-hub==1.5.0, datasets==2.15.0) which were chosen for mutual compatibility.
#   - No parent package in the image transitively pins diffusers; it is a direct top-level pin.
#   - Bumping the requirements.txt pin from 0.24.0 to 0.38.0 is a 14-minor-version jump that
#     would pull a new transformers/huggingface_hub/accelerate floor and risks breaking the
#     rest of the pinned HF stack, which is out of scope for a CVE-only patch.
# Using --no-deps to keep the existing HF stack intact; diffusers 0.38.0 runtime imports are
# satisfied by the already-installed transformers/accelerate/huggingface_hub/safetensors.
RUN pip install --no-cache-dir --no-deps 'diffusers==0.38.0'

# Override urllib3 2.6.3 -> 2.7.0 in both envs to fix GHSA-qccp-gfcp-xxvc (CVE-2026-44431)
# and GHSA-mf9v-mfxr-j63j (CVE-2026-44432).
# Root cause (verified 2026-05 via PyPI metadata against the built image):
#   - urllib3 is a transitive dep brought in by requests, botocore, azureml-core, etc.;
#     none of these parents pin urllib3 to a version that excludes 2.7.0 (requests allows
#     urllib3<3, botocore allows urllib3<3 on py>=3.10), so there is no parent we can bump
#     to pick up 2.7.0 automatically.
#   - The upstream PTCA base image still ships 2.6.3 in both /opt/conda (py3.13) and
#     /opt/conda/envs/ptca (py3.10); the parent packages above resolve to 2.6.3 because
#     that is the highest version cached/mirrored at base-image build time.
# Bound the upgrade to the 2.x line to keep ABI-compatible with the existing requests stack.
RUN pip install --no-cache-dir --upgrade 'urllib3>=2.7.0,<3' \
 && /opt/conda/bin/python3.13 -m pip install --no-cache-dir --upgrade 'urllib3>=2.7.0,<3'

RUN conda clean -a -y && rm -rf /opt/miniconda/pkgs/
